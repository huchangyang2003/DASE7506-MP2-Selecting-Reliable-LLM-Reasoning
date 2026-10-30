"""Nine predeclared, matched objective fits; validation-only comparison."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'

import numpy as np

from common import DATA, PROTOCOL, check_data, load_bank, parse_answer, read_jsonl, validate_choices
from no_test_access import install_guard

ROOT = Path(__file__).resolve().parent
OBJECTIVES = ('bce', 'pairwise', 'listwise')
SEEDS = (20261022, 20261023, 20261024)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def run(arguments, log):
    tick = time.perf_counter()
    result = subprocess.run([sys.executable, '-B', *arguments], cwd=ROOT,
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                            capture_output=True, text=True, encoding='utf-8', timeout=180)
    elapsed = time.perf_counter() - tick
    log.write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        save(log.with_suffix('.failure.json'), {'arguments': arguments, 'wall_seconds': elapsed,
                                               'return_code': result.returncode})
        raise RuntimeError(result.stderr)
    return elapsed


def summary(correct, hard):
    return {'correct': int(correct.sum()), 'n': len(correct), 'accuracy': 100 * float(correct.mean()),
            'disagreement_correct': int(correct[hard].sum()), 'disagreement_n': int(hard.sum()),
            'disagreement_accuracy': 100 * float(correct[hard].mean())}


def main():
    install_guard()
    destination = ROOT / 'runs/stage1'
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'summary.json').exists():
        raise FileExistsError('Stage already completed; preserve its results and use a new output tree for repeats')
    budget = {'declared_at_utc': datetime.now(timezone.utc).isoformat(), 'objectives': OBJECTIVES,
              'seeds': SEEDS, 'fits': 9, 'hidden': 32, 'epochs': 60, 'batch_questions': 128,
              'learning_rate': 0.001, 'weight_l2': 0.0001, 'features': 'unchanged V1 group features',
              'protocol': PROTOCOL, 'training_split': 'train', 'evaluation_split': 'validation',
              'test_evaluation': False, 'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ('train_ranker.py', 'objectives.py', 'features.py', 'run_stage1.py', 'methods/hybrid_model.py')},
              'plan_sha256': hashlib.sha256((ROOT.parent / 'PLAN.md').read_bytes()).hexdigest(),
              'selection': 'Highest mean disagreement accuracy across matched seeds, then mean full accuracy',
              'promising_rule': 'Positive mean disagreement gain over matched BCE and gain on at least two of three seeds'}
    save(destination / 'budget.json', budget)
    check_data(DATA, splits=('train', 'validation'))
    outputs, records = {}, []
    for seed in SEEDS:
        seed_hashes = []
        for objective in OBJECTIVES:
            name = f'{objective}_seed{seed}'
            folder = destination / name
            folder.mkdir(exist_ok=True)
            training_wall = run(['train_ranker.py', '--objective', objective, '--seed', str(seed),
                                 '--output-dir', str(folder)], folder / 'training.log')
            prediction_wall = run(['run_official.py', 'predict.py', '--method', 'submission.py',
                                   '--checkpoint', str(folder / 'checkpoint'), '--split', 'validation',
                                   '--output', str(folder / 'validation-predictions.json')], folder / 'prediction.log')
            scoring_wall = run(['run_official.py', 'score_predictions.py', '--split', 'validation',
                                '--predictions', str(folder / 'validation-predictions.json'),
                                '--output', str(folder / 'validation-score.json')], folder / 'scoring.log')
            training = json.loads((folder / 'training.json').read_text())
            score = json.loads((folder / 'validation-score.json').read_text())
            outputs[name] = json.loads((folder / 'validation-predictions.json').read_text())
            seed_hashes.append((training['training']['initial_parameter_sha256'], training['training']['question_shuffle_sha256']))
            record = {'method': name, 'objective': objective, 'seed': seed, 'training_wall_seconds': training_wall,
                      'prediction_wall_seconds': prediction_wall, 'scoring_wall_seconds': scoring_wall,
                      'total_wall_seconds': prediction_wall + scoring_wall,
                      'inference_uncompressed_bytes': training['inference_uncompressed_bytes'],
                      'official_correct': score['correct'], 'official_accuracy': score['accuracy']}
            records.append(record)
            print(json.dumps(record), flush=True)
        if len(set(seed_hashes)) != 1:
            raise AssertionError('Objective runs did not share initialization and question batches')
    bank = load_bank('validation')
    refs = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    if len(refs) != len(bank) or any(row['id'] != ref['id'] or row['question'] != ref['question']
                                    for row, ref in zip(bank, refs)):
        raise ValueError('Validation alignment mismatch')
    answers = [[parse_answer(candidate['text']) for candidate in row['candidates']] for row in bank]
    labels = np.asarray([[answer is not None and answer == parse_answer(ref['answer']) for answer in group]
                         for group, ref in zip(answers, refs)])
    hard = np.array([len(set(group) - {None}) >= 2 for group in answers])
    correctness = {}
    for record in records:
        prediction = outputs[record['method']]
        validate_choices(bank, prediction['ids'], prediction['selected'])
        correct = labels[np.arange(len(bank)), prediction['selected']]
        correctness[record['method']] = correct
        record.update(summary(correct, hard))
        assert record['correct'] == record['official_correct']
    aggregates, comparisons = [], {}
    rng = np.random.default_rng(20261022)
    for objective in OBJECTIVES:
        chosen = [record for record in records if record['objective'] == objective]
        aggregates.append({'objective': objective, 'seeds': 3,
                           'accuracy_mean': float(np.mean([row['accuracy'] for row in chosen])),
                           'accuracy_sample_std': float(np.std([row['accuracy'] for row in chosen], ddof=1)),
                           'disagreement_mean': float(np.mean([row['disagreement_accuracy'] for row in chosen])),
                           'disagreement_sample_std': float(np.std([row['disagreement_accuracy'] for row in chosen], ddof=1)),
                           'total_training_wall_seconds': sum(row['training_wall_seconds'] for row in chosen)})
        if objective == 'bce':
            continue
        deltas = np.stack([correctness[f'{objective}_seed{seed}'].astype(float)
                           - correctness[f'bce_seed{seed}'].astype(float) for seed in SEEDS])
        differences = {}
        for subset, mask in (('all', np.ones(len(bank), dtype=bool)), ('disagreement', hard)):
            question_delta = deltas[:, mask].mean(axis=0)
            boots = np.concatenate([question_delta[rng.integers(0, len(question_delta), size=(500, len(question_delta)))].mean(axis=1)
                                    for _ in range(10)]) * 100
            differences[subset] = {'mean_gain_percentage_points': float(question_delta.mean() * 100),
                                   'gain_by_seed_percentage_points': (deltas[:, mask].mean(axis=1) * 100).tolist(),
                                   'wins_across_three_seeds': int(np.sum(deltas[:, mask].mean(axis=1) > 0)),
                                   'question_bootstrap_95_interval': np.quantile(boots, [0.025, 0.975]).tolist()}
        comparisons[objective] = differences
    winner = max(aggregates, key=lambda row: (row['disagreement_mean'], row['accuracy_mean']))
    best = winner['objective']
    promising = (best != 'bce' and comparisons[best]['disagreement']['mean_gain_percentage_points'] > 0
                 and comparisons[best]['disagreement']['wins_across_three_seeds'] >= 2)
    selection = {'best_average_objective': best, 'promising_under_predeclared_rule': promising,
                 'primary_seed': 20261022, 'development_checkpoint': f'runs/stage1/{best}_seed20261022/checkpoint',
                 'not_a_final_submission_or_test_result': True,
                 'test_accessed_for_training_or_evaluation': False,
                 'note': 'Seed-averaged results summarize repeated single models, not an inference ensemble. '
                         'V1 public test was already seen before this development; intervals are descriptive on reused validation.'}
    save(destination / 'summary.json', records)
    save(destination / 'aggregates.json', aggregates)
    save(destination / 'paired_comparisons.json', comparisons)
    save(destination / 'selection.json', selection)
    save(destination / 'environment.json', {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
              'thread_limits': {key: os.environ[key] for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
              'matching_initializations_and_shuffles': True, 'test_data_absent': not (DATA / 'gsm8k_test.jsonl').exists()})
    with (destination / 'experiments.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    save(destination / 'validation_diagnostics.json', [
        {'id': row['id'], 'disagreement': bool(hard[i]),
         'correct': {name: bool(value[i]) for name, value in correctness.items()}}
        for i, row in enumerate(bank)])
    print(json.dumps({'aggregates': aggregates, 'comparisons': comparisons, 'selection': selection}, indent=2))


if __name__ == '__main__':
    main()
