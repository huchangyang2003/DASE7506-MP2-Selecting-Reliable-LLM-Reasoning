"""Evaluate the fixed fusion budget with the unchanged official validation scorer."""
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import numpy as np
import numpy.testing  # Preload assertion helpers before the inference read guard.
from common import DATA, load_inputs, parse_answer, read_jsonl
from fusion_core import answer_statistics
from methods.fusion_model import predict, candidate_scores
from no_test_access import install_guard
from run_stage1 import ROOT, SEEDS, run, save, summary
from weighted_bce import disagreement_mask


def main():
    install_guard()
    destination = ROOT / 'runs/stage3'
    if (destination / 'summary.json').exists():
        raise FileExistsError('Stage 3 results already exist')
    training = json.loads((destination / 'training_complete.json').read_text())
    assert training['no_validation_or_test_opened']
    save(destination / 'evaluation_started.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
                                                   'training_finished_at_utc': training['finished_at_utc']})
    bank, arrays = load_inputs('validation')
    refs = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    assert len(refs) == len(bank) and all(b['id'] == ref['id'] and b['question'] == ref['question'] for b, ref in zip(bank, refs))
    labels = np.array([[int((a := parse_answer(c['text'])) is not None and a == parse_answer(ref['answer']))
                        for c in b['candidates']] for b, ref in zip(bank, refs)], dtype=bool)
    hard = disagreement_mask(bank)
    correctness, choices, records = {}, {}, []
    modes = []
    for model in training['models']:
        mode, seed = model['mode'], model['seed']
        if mode not in modes:
            modes.append(mode)
        name = f'{mode}_seed{seed}'
        folder = destination / name
        checkpoint = ROOT / model['checkpoint']
        prediction_wall = run(['run_official.py', 'predict.py', '--method', 'methods/fusion_model.py',
                               '--checkpoint', str(checkpoint), '--split', 'validation',
                               '--output', str(folder / 'validation-predictions.json')], folder / 'prediction.log')
        score_wall = run(['run_official.py', 'score_predictions.py', '--split', 'validation',
                          '--predictions', str(folder / 'validation-predictions.json'),
                          '--output', str(folder / 'validation-score.json')], folder / 'scoring.log')
        output = json.loads((folder / 'validation-predictions.json').read_text())
        score = json.loads((folder / 'validation-score.json').read_text())
        picked = np.asarray(output['selected'])
        correct = labels[np.arange(len(bank)), picked]
        assert int(correct.sum()) == score['correct']
        match = None
        if mode == 'mlp_control':
            old = json.loads((ROOT / f'runs/stage1/bce_seed{seed}/validation-predictions.json').read_text())
            match = output['selected'] == old['selected']
            assert match, 'Fusion MLP control differs from historical matched control'
        choices[name], correctness[name] = picked, correct
        records.append({'mode': mode, 'seed': seed, **summary(correct, hard),
                         'other_correct': int(correct[~hard].sum()), 'other_n': int((~hard).sum()),
                         'prediction_wall_seconds': prediction_wall, 'scoring_wall_seconds': score_wall,
                         'total_wall_seconds': prediction_wall + score_wall,
                         'inference_uncompressed_bytes': model['inference_uncompressed_bytes'],
                         'exactly_matches_historical_control': match})
        print(json.dumps(records[-1]), flush=True)
    aggregates, comparisons = [], {}
    rng = np.random.default_rng(20261022)
    for mode in modes:
        group = [r for r in records if r['mode'] == mode]
        aggregates.append({'mode': mode, 'seeds': 3,
                           'accuracy_mean': float(np.mean([r['accuracy'] for r in group])),
                           'accuracy_sample_std': float(np.std([r['accuracy'] for r in group], ddof=1)),
                           'disagreement_mean': float(np.mean([r['disagreement_accuracy'] for r in group])),
                           'disagreement_sample_std': float(np.std([r['disagreement_accuracy'] for r in group], ddof=1))})
        if mode == 'mlp_control':
            continue
        differences = np.stack([correctness[f'{mode}_seed{s}'].astype(float) -
                                 correctness[f'mlp_control_seed{s}'].astype(float) for s in SEEDS])
        comparisons[mode] = {}
        for subset, mask in [('all', np.ones(len(bank), bool)), ('disagreement', hard), ('other', ~hard)]:
            per_question = differences[:, mask].mean(axis=0)
            boots = np.concatenate([per_question[rng.integers(0, len(per_question), (500, len(per_question)))].mean(axis=1)
                                    for _ in range(10)]) * 100
            comparisons[mode][subset] = {
                'mean_gain_percentage_points': float(per_question.mean() * 100),
                'gain_by_seed_percentage_points': (differences[:, mask].mean(axis=1) * 100).tolist(),
                'seed_wins': int(np.sum(differences[:, mask].mean(axis=1) > 0)),
                'question_bootstrap_95_interval': np.quantile(boots, [0.025, 0.975]).tolist(),
                'paired_counts_by_seed': [{'seed': s,
                    'corrected': int(np.sum((differences[i] > 0) & mask)),
                    'broken': int(np.sum((differences[i] < 0) & mask))} for i, s in enumerate(SEEDS)]}
    winner = max(aggregates, key=lambda r: (r['disagreement_mean'], r['accuracy_mean'], -modes.index(r['mode'])))
    mode = winner['mode']
    practical = (mode != 'mlp_control' and comparisons[mode]['disagreement']['mean_gain_percentage_points'] >= 1
                 and comparisons[mode]['disagreement']['seed_wins'] >= 2)
    selection = {'best_average_mode': mode, 'passes_minimum_practical_candidate_rule': practical,
                 'primary_seed': 20261022, 'development_checkpoint': f'runs/stage3/{mode}_seed20261022/checkpoint',
                 'test_evaluated': False, 'v1_replaced': False,
                 'note': 'Repeated validation development; bootstrap intervals descriptive, no selection correction. '
                         'Mean across seeds is not an inference ensemble; never select best seed.'}
    save(destination / 'selection.json', selection)
    save(destination / 'summary.json', records)
    save(destination / 'aggregates.json', aggregates)
    save(destination / 'paired_comparisons.json', comparisons)
    with (destination / 'experiments.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    save(destination / 'validation_diagnostics.json', [
        {'id': b['id'], 'disagreement': bool(hard[i]),
         'correct': {name: bool(a[i]) for name, a in correctness.items()}} for i, b in enumerate(bank)])
    support, _, _ = answer_statistics(bank)
    vote_picks = support.argmax(axis=1)
    vote_correct = labels[np.arange(len(bank)), vote_picks]
    # Diagnostic oracle uses validation gold to measure headroom; never used by inference.
    complementarity = []
    for seed in SEEDS:
        cp = destination / f'mlp_control_seed{seed}/checkpoint'
        with np.load(cp / 'model.npz', allow_pickle=False) as m:
            linear = ((np.asarray(arrays['embeddings'], float) - m['emb_mean']) / m['emb_scale']) @ m['emb_weights'] + m['emb_bias'].item()
        lin_correct = labels[np.arange(len(bank)), linear.argmax(axis=1)]
        mlp_correct = correctness[f'mlp_control_seed{seed}']
        union = mlp_correct | lin_correct | vote_correct
        complementarity.append({'seed': seed, 'embedding': summary(lin_correct, hard),
                                 'vote': summary(vote_correct, hard), 'oracle_union': summary(union, hard),
                                 'note': 'Gold-dependent diagnostic upper bound, not an implementable selector or result'})
    save(destination / 'complementarity.json', complementarity)
    counts = np.asarray([len({a for c in b['candidates'] if (a := parse_answer(c['text'])) is not None}) for b in bank])
    save(destination / 'strata.json', [{'distinct_valid_answers': int(c), 'n': int(np.sum(counts == c)),
              **{m: 100 * float(np.mean([correctness[f'{m}_seed{s}'][counts == c].mean() for s in SEEDS])) for m in modes}}
                                     for c in sorted(set(counts))])
    verify(bank, arrays, destination / f'{mode}_seed20261022/checkpoint',
           choices[f'{mode}_seed20261022'], destination)
    print(json.dumps({'aggregates': aggregates, 'selection': selection}, indent=2), flush=True)


def verify(bank, arrays, checkpoint, official, destination):
    import copy
    active = [False]
    opened = []
    allowed = {(checkpoint / name).resolve() for name in ('model.npz', 'config.json')}
    def audit(event, args):
        if not active[0]:
            return
        if event.startswith('socket.'):
            raise PermissionError('Offline inference only')
        if event == 'open' and isinstance(args[0], str):
            p = Path(args[0]).resolve()
            if p not in allowed:
                raise PermissionError(f'Inference read outside checkpoint: {p}')
            opened.append(str(p))
    sys.addaudithook(audit)
    active[0] = True
    reloaded = predict(bank, arrays, checkpoint)
    assert reloaded == official.tolist()
    short = copy.deepcopy(bank[:17])
    for i, row in enumerate(short):
        row['id'] = f'new_unseen_id_{i}'
    short_arrays = {k: a[:17] for k, a in arrays.items()}
    assert predict(short, short_arrays, checkpoint) == reloaded[:17]
    order = np.array([2, 0, 3, 1])
    permuted = copy.deepcopy(bank)
    for row in permuted:
        row['candidates'] = [{**row['candidates'][j], 'index': i} for i, j in enumerate(order)]
    new_scores = candidate_scores(permuted, {k: a[:, order] for k, a in arrays.items()}, checkpoint)
    old_scores = candidate_scores(bank, arrays, checkpoint)
    np.testing.assert_allclose(new_scores, old_scores[:, order], rtol=1e-5, atol=1e-5)
    mapped = order[new_scores.argmax(axis=1)]
    index_changes = int(np.sum(mapped != official))
    answer_changes = sum(parse_answer(row['candidates'][int(a)]['text']) != parse_answer(row['candidates'][int(b)]['text'])
                         for row, a, b in zip(bank, mapped, official))
    active[0] = False
    save(destination / 'inference_verification.json', {
        'reload_matches_all_official_choices': True, 'new_ids_17_rows_matches': True,
        'candidate_permutation_scores_allclose': True, 'permutation_index_changes': index_changes,
        'permutation_parsed_answer_changes': answer_changes,
        'read_only_checkpoint_files': sorted(set(opened)), 'socket_attempts_blocked_by_guard': True,
        'no_dataset_lookup_during_inference': True})


if __name__ == '__main__':
    if sys.argv[1:] == ['--verify-only']:
        install_guard()
        destination = ROOT / 'runs/stage3'
        selection = json.loads((destination / 'selection.json').read_text())
        folder = ROOT / selection['development_checkpoint']
        official = json.loads((folder.parent / 'validation-predictions.json').read_text())
        bank, arrays = load_inputs('validation')
        verify(bank, arrays, folder, np.asarray(official['selected']), destination)
        print('Selected checkpoint inference verification passed; no refit or rescoring.')
    else:
        main()
