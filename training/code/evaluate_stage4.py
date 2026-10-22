"""Nine predeclared validation runs, after every arithmetic fusion fit finished."""
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '4'
import numpy as np
from arithmetic_features import build_arithmetic_features
from arithmetic_fusion_core import MODES
from common import DATA, load_inputs, parse_answer, read_jsonl
from methods import arithmetic_fusion
from no_test_access import install_guard
from run_stage1 import ROOT, SEEDS, run, save, summary
from verify_input_only import verify
from weighted_bce import disagreement_mask


def comparison(correctness, mode, reference, hard):
    differences = np.stack([correctness[f'{mode}_seed{s}'].astype(float) -
                             correctness[f'{reference}_seed{s}'].astype(float) for s in SEEDS])
    rng = np.random.default_rng(20261022)
    result = {}
    for name, mask in [('all', np.ones(len(hard), bool)), ('disagreement', hard), ('other', ~hard)]:
        delta = differences[:, mask].mean(axis=0)
        boots = np.concatenate([delta[rng.integers(0, len(delta), (500, len(delta)))].mean(axis=1)
                                for _ in range(10)]) * 100
        result[name] = {'mean_gain_percentage_points': float(delta.mean()*100),
            'gain_by_seed_percentage_points': (differences[:, mask].mean(axis=1)*100).tolist(),
            'seed_wins': int(np.sum(differences[:, mask].mean(axis=1) > 0)),
            'question_bootstrap_95_interval': np.quantile(boots, [0.025, 0.975]).tolist(),
            'paired_by_seed': [{'seed': seed, 'corrected': int(np.sum((differences[i] > 0) & mask)),
                               'broken': int(np.sum((differences[i] < 0) & mask))} for i, seed in enumerate(SEEDS)]}
    return result


def main():
    install_guard()
    destination = ROOT / 'runs/stage4_final'
    if (destination / 'summary.json').exists(): raise FileExistsError('Do not overwrite a completed stage')
    training = json.loads((destination / 'training_complete.json').read_text())
    assert training['no_validation_or_test_opened']
    assert training['parser_sha256'] == hashlib.sha256((ROOT / 'arithmetic_features.py').read_bytes()).hexdigest()
    save(destination / 'evaluation_started.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'training_finished_at_utc': training['finished_at_utc'], 'sources_before_validation_sha256': {
            n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
            ('evaluate_stage4.py', 'verify_input_only.py', 'arithmetic_features.py', 'methods/arithmetic_fusion.py')}})
    bank, arrays = load_inputs('validation')
    refs = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    assert len(bank) == len(refs) and all(b['id'] == r['id'] and b['question'] == r['question'] for b, r in zip(bank, refs))
    labels = np.array([[int((a := parse_answer(c['text'])) is not None and a == parse_answer(ref['answer']))
                       for c in b['candidates']] for b, ref in zip(bank, refs)], dtype=bool)
    hard = disagreement_mask(bank)
    records, correctness, choices = [], {}, {}
    for model in training['models']:
        mode, seed = model['mode'], model['seed']
        name = f'{mode}_seed{seed}'; folder = destination / name
        checkpoint = ROOT / model['checkpoint']
        prediction_seconds = run(['run_official.py', 'predict.py', '--method', 'methods/arithmetic_fusion.py',
            '--checkpoint', str(checkpoint), '--split', 'validation', '--output', str(folder / 'validation-predictions.json')],
            folder / 'prediction.log')
        scoring_seconds = run(['run_official.py', 'score_predictions.py', '--split', 'validation',
            '--predictions', str(folder / 'validation-predictions.json'), '--output', str(folder / 'validation-score.json')],
            folder / 'scoring.log')
        prediction = json.loads((folder / 'validation-predictions.json').read_text())
        score = json.loads((folder / 'validation-score.json').read_text())
        chosen = np.asarray(prediction['selected'])
        correct = labels[np.arange(len(bank)), chosen]
        assert int(correct.sum()) == score['correct']
        exact_match = None
        if mode == 'control':
            old = json.loads((ROOT / f'runs/stage3/stacking_0.001_seed{seed}/validation-predictions.json').read_text())
            exact_match = old['selected'] == prediction['selected']; assert exact_match
        choices[name], correctness[name] = chosen, correct
        record = {'mode': mode, 'seed': seed, **summary(correct, hard),
                  'other_correct': int(correct[~hard].sum()), 'other_n': int((~hard).sum()),
                  'prediction_wall_seconds': prediction_seconds, 'scoring_wall_seconds': scoring_seconds,
                  'total_wall_seconds': prediction_seconds + scoring_seconds,
                  'inference_uncompressed_bytes': model['inference_uncompressed_bytes'],
                  'control_matches_stage3': exact_match}
        records.append(record); print(json.dumps(record), flush=True)
    aggregates = [{'mode': m, 'seeds': 3,
                   'accuracy_mean': float(np.mean([r['accuracy'] for r in records if r['mode'] == m])),
                   'accuracy_sample_std': float(np.std([r['accuracy'] for r in records if r['mode'] == m],ddof=1)),
                   'disagreement_mean': float(np.mean([r['disagreement_accuracy'] for r in records if r['mode'] == m])),
                   'disagreement_sample_std': float(np.std([r['disagreement_accuracy'] for r in records if r['mode'] == m],ddof=1))}
                  for m in MODES]
    comparisons = {f'{m}_vs_{ref}': comparison(correctness, m, ref, hard)
                   for m, ref in [('coverage', 'control'), ('arithmetic', 'control'), ('arithmetic', 'coverage')]}
    winner = max(aggregates, key=lambda r: (r['disagreement_mean'], r['accuracy_mean'], -MODES.index(r['mode'])))
    best = winner['mode']
    practical = best != 'control' and comparisons[f'{best}_vs_control']['disagreement']['mean_gain_percentage_points'] >= 1 and comparisons[f'{best}_vs_control']['disagreement']['seed_wins'] >= 2
    selection = {'best_average_mode': best, 'passes_minimum_practical_candidate_rule': practical,
                 'primary_seed': 20261022, 'development_checkpoint': f'runs/stage4_final/{best}_seed20261022/checkpoint',
                 'test_evaluated': False, 'v1_or_stage3_replaced': False,
                 'arithmetic_extra_vs_coverage': comparisons['arithmetic_vs_coverage']['disagreement'],
                 'note': 'Development on repeatedly used validation, not a new blind test. Means across seeds are not ensembles.'}
    save(destination / 'selection.json', selection)
    save(destination / 'summary.json', records); save(destination / 'aggregates.json', aggregates)
    save(destination / 'paired_comparisons.json', comparisons)
    with (destination / 'experiments.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    features, names, details = build_arithmetic_features(bank, details=True)
    checked = np.asarray([[c['checked'] for c in g] for g in details])
    bad = np.asarray([[c['bad'] for c in g] for g in details])
    covered = checked > 0
    save(destination / 'parser_validation_summary.json', {'candidate_count': int(covered.size),
        'covered_candidates': int(covered.sum()), 'coverage_percent': float(100*covered.mean()),
        'equals': sum(c['equals'] for g in details for c in g), 'checked_equations': int(checked.sum()),
        'bad_equations': int(bad.sum()), 'candidates_with_bad_equation': int(np.sum(bad>0)),
        'candidate_correctness_given_any_bad_equation_percent': float(100*labels[bad>0].mean()) if (bad>0).any() else None,
        'candidate_correctness_without_bad_equation_percent': float(100*labels[bad==0].mean()),
        'interpretation': 'Scorer-label association, not proof of reasoning correctness or a hard veto rule.'})
    counts = np.asarray([len({a for c in b['candidates'] if (a:=parse_answer(c['text'])) is not None}) for b in bank])
    save(destination / 'strata.json', [{'distinct_valid_answers': int(c), 'n': int(np.sum(counts==c)),
        **{m: 100*float(np.mean([correctness[f'{m}_seed{s}'][counts==c].mean() for s in SEEDS])) for m in MODES}}
        for c in sorted(set(counts))])
    subsets = {'all_four_covered': covered.all(axis=1), 'some_covered': covered.any(axis=1)&~covered.all(axis=1),
               'none_covered': ~covered.any(axis=1), 'some_candidate_has_bad': (bad>0).any(axis=1)}
    save(destination / 'coverage_strata.json', [{'subset': n, 'n': int(mask.sum()),
        **{m: 100*float(np.mean([correctness[f'{m}_seed{s}'][mask].mean() for s in SEEDS])) if mask.any() else None for m in MODES}}
        for n, mask in subsets.items()])
    save(destination / 'validation_diagnostics.json', [{'id': row['id'], 'disagreement': bool(hard[i]),
        'checked_equations_by_candidate': checked[i].tolist(), 'bad_equations_by_candidate': bad[i].tolist(),
        'correct': {n: bool(v[i]) for n,v in correctness.items()}} for i,row in enumerate(bank)])
    examples = []
    for q,row in enumerate(bank):
        old, new = int(choices['control_seed20261022'][q]), int(choices['arithmetic_seed20261022'][q])
        if old != new:
            examples.append({'id': row['id'], 'question': row['question'], 'control_index': old, 'arithmetic_index': new,
                'control_correct': bool(labels[q,old]), 'arithmetic_correct': bool(labels[q,new]),
                'control_text': row['candidates'][old]['text'], 'arithmetic_text': row['candidates'][new]['text'],
                'control_parser': details[q][old], 'arithmetic_parser': details[q][new]})
    save(destination / 'changed_choice_examples.json', examples)
    # Verify actual arithmetic code even if it loses selection, plus the selected mode.
    checks = {}
    for mode in dict.fromkeys(('arithmetic', best)):
        cp = destination / f'{mode}_seed20261022/checkpoint'
        checks[mode] = verify(arithmetic_fusion, bank, arrays, cp, choices[f'{mode}_seed20261022'])
    save(destination / 'inference_verification.json', checks)
    save(destination / 'completion.json', {'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'final_fusion_fits': 9, 'official_validation_runs': len(records), 'test_evaluated': False,
        'verification_passed': True})
    print(json.dumps({'aggregates': aggregates, 'selection': selection},indent=2),flush=True)


if __name__ == '__main__': main()
