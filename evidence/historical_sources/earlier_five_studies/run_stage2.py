"""Predeclared weighting experiment: four modes, three matched seeds, no test."""

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from common import DATA, check_data, load_bank, parse_answer, read_jsonl, validate_choices
from no_test_access import install_guard
from run_stage1 import ROOT, SEEDS, run, save, summary
from weighted_bce import MODES, disagreement_mask


def main():
    install_guard()
    destination = ROOT / 'runs/stage2'
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'summary.json').exists():
        raise FileExistsError('Stage 2 already complete; preserve recorded results')
    save(destination / 'budget.json', {
        'declared_at_utc': datetime.now(timezone.utc).isoformat(), 'weight_modes': MODES, 'seeds': SEEDS,
        'fits': 12, 'objective': 'BCE', 'hidden': 32, 'epochs': 60, 'batch_questions': 128,
        'learning_rate': 0.001, 'weight_l2': 0.0001,
        'normalization': 'Each mode divided by full TRAIN raw-weight mean, not batch mean',
        'standardization': 'Identical full-training mean and scale in all modes',
        'no_test_evaluation': True, 'no_additional_feature_capacity_or_loss_search': True,
        'selection': 'Highest mean disagreement accuracy, then mean complete accuracy; never pick the best seed',
        'minimum_practical_candidate_rule': 'Mean disagreement gain >=1 percentage point and >=2 of 3 seed wins; '
                                            'bootstrap interval spanning zero remains uncertain evidence',
        'plan_sha256': hashlib.sha256((ROOT.parent / 'STAGE2_PLAN.md').read_bytes()).hexdigest(),
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                          ('weighted_bce.py', 'train_weighted.py', 'run_stage2.py', 'objectives.py',
                           'features.py', 'methods/hybrid_model.py')}})
    check_data(DATA, splits=('train', 'validation'))
    records, outputs = [], {}
    for seed in SEEDS:
        hashes = []
        for mode in MODES:
            name = f'{mode}_seed{seed}'
            folder = destination / name
            folder.mkdir(exist_ok=True)
            training_wall = run(['train_weighted.py', '--weight-mode', mode, '--seed', str(seed),
                                 '--output-dir', str(folder)], folder / 'training.log')
            training = json.loads((folder / 'training.json').read_text())
            hashes.append((training['training']['initial_parameter_sha256'], training['training']['question_shuffle_sha256']))
            historical_control_match = None
            if mode == 'uniform':
                with np.load(folder / 'checkpoint/model.npz', allow_pickle=False) as current:
                    with np.load(ROOT / f'runs/stage1/bce_seed{seed}/checkpoint/model.npz', allow_pickle=False) as previous:
                        historical_control_match = all(np.array_equal(current[key], previous[key]) for key in previous.files)
                if not historical_control_match:
                    raise AssertionError('Uniform control differs from stage-1 BCE parameters')
            prediction_wall = run(['run_official.py', 'predict.py', '--method', 'submission.py',
                                   '--checkpoint', str(folder / 'checkpoint'), '--split', 'validation',
                                   '--output', str(folder / 'validation-predictions.json')], folder / 'prediction.log')
            scoring_wall = run(['run_official.py', 'score_predictions.py', '--split', 'validation',
                                '--predictions', str(folder / 'validation-predictions.json'),
                                '--output', str(folder / 'validation-score.json')], folder / 'scoring.log')
            score = json.loads((folder / 'validation-score.json').read_text())
            outputs[name] = json.loads((folder / 'validation-predictions.json').read_text())
            record = {'method': name, 'weight_mode': mode, 'seed': seed, 'training_wall_seconds': training_wall,
                      'prediction_wall_seconds': prediction_wall, 'scoring_wall_seconds': scoring_wall,
                      'total_wall_seconds': prediction_wall + scoring_wall,
                      'inference_uncompressed_bytes': training['inference_uncompressed_bytes'],
                      'hard_fraction_of_training_loss_weight': training['question_weights']['hard_fraction_of_total_weight'],
                      'uniform_parameters_exactly_match_stage1': historical_control_match,
                      'official_correct': score['correct'], 'official_accuracy': score['accuracy']}
            records.append(record)
            print(json.dumps(record), flush=True)
        if len(set(hashes)) != 1:
            raise AssertionError('Initialization or question schedule differs across modes')
    bank = load_bank('validation')
    refs = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    if len(refs) != len(bank) or any(row['id'] != ref['id'] or row['question'] != ref['question']
                                    for row, ref in zip(bank, refs)):
        raise ValueError('Validation alignment mismatch')
    labels = np.asarray([[int((answer := parse_answer(candidate['text'])) is not None
                             and answer == parse_answer(ref['answer']))
                          for candidate in row['candidates']] for row, ref in zip(bank, refs)], dtype=bool)
    hard = disagreement_mask(bank)
    correctness = {}
    for row in records:
        pred = outputs[row['method']]
        validate_choices(bank, pred['ids'], pred['selected'])
        correct = labels[np.arange(len(bank)), pred['selected']]
        correctness[row['method']] = correct
        row.update(summary(correct, hard))
        row['other_n'] = int((~hard).sum())
        row['other_correct'] = int(correct[~hard].sum())
        row['other_accuracy'] = 100 * float(correct[~hard].mean())
        assert row['correct'] == row['official_correct']
    aggregates, comparisons = [], {}
    rng = np.random.default_rng(20261022)
    for mode in MODES:
        group = [row for row in records if row['weight_mode'] == mode]
        aggregates.append({'weight_mode': mode, 'seeds': 3,
                           'accuracy_mean': float(np.mean([row['accuracy'] for row in group])),
                           'accuracy_sample_std': float(np.std([row['accuracy'] for row in group], ddof=1)),
                           'disagreement_mean': float(np.mean([row['disagreement_accuracy'] for row in group])),
                           'disagreement_sample_std': float(np.std([row['disagreement_accuracy'] for row in group], ddof=1)),
                           'other_accuracy_mean': float(np.mean([row['other_accuracy'] for row in group])),
                           'total_training_wall_seconds': sum(row['training_wall_seconds'] for row in group)})
        if mode == 'uniform':
            continue
        differences = np.stack([correctness[f'{mode}_seed{seed}'].astype(float)
                                - correctness[f'uniform_seed{seed}'].astype(float) for seed in SEEDS])
        comparisons[mode] = {}
        for subset, mask in (('all', np.ones(len(bank), dtype=bool)), ('disagreement', hard), ('other', ~hard)):
            delta = differences[:, mask].mean(axis=0)
            boots = np.concatenate([delta[rng.integers(0, len(delta), size=(500, len(delta)))].mean(axis=1)
                                    for _ in range(10)]) * 100
            comparisons[mode][subset] = {
                'mean_gain_percentage_points': float(delta.mean() * 100),
                'gain_by_seed_percentage_points': (differences[:, mask].mean(axis=1) * 100).tolist(),
                'seed_wins': int(np.sum(differences[:, mask].mean(axis=1) > 0)),
                'question_bootstrap_95_interval': np.quantile(boots, [0.025, 0.975]).tolist()}
    winner = max(aggregates, key=lambda row: (row['disagreement_mean'], row['accuracy_mean']))
    mode = winner['weight_mode']
    minimum_rule = (mode != 'uniform' and comparisons[mode]['disagreement']['mean_gain_percentage_points'] >= 1
                    and comparisons[mode]['disagreement']['seed_wins'] >= 2)
    save(destination / 'summary.json', records)
    save(destination / 'aggregates.json', aggregates)
    save(destination / 'paired_comparisons.json', comparisons)
    save(destination / 'selection.json', {'best_average_weight_mode': mode,
              'passes_minimum_practical_candidate_rule': minimum_rule, 'primary_seed': 20261022,
              'development_checkpoint': f'runs/stage2/{mode}_seed20261022/checkpoint',
              'test_evaluated': False, 'not_a_final_submission_or_independent_test': True})
    with (destination / 'experiments.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    save(destination / 'validation_diagnostics.json', [
        {'id': row['id'], 'disagreement': bool(hard[i]),
         'correct': {name: bool(values[i]) for name, values in correctness.items()}}
        for i, row in enumerate(bank)])
    counts = np.array([len({answer for candidate in row['candidates']
                           if (answer := parse_answer(candidate['text'])) is not None}) for row in bank])
    save(destination / 'strata.json', [
        {'distinct_valid_answers': int(count), 'n': int((counts == count).sum()),
         **{mode: 100 * float(np.mean([correctness[f'{mode}_seed{seed}'][counts == count].mean() for seed in SEEDS]))
            for mode in MODES}} for count in sorted(set(counts))])
    print(json.dumps({'aggregates': aggregates, 'comparisons': comparisons,
                      'best_mode': mode, 'passes_minimum_rule': minimum_rule}, indent=2))


if __name__ == '__main__':
    main()
