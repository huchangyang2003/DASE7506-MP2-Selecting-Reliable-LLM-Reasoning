"""Fixed stage-B 33-fit experiment; TRAIN only, no validation or test scoring."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
import zipfile
import numpy as np
import stage_a as io  # Existing TRAIN-only access guard and immutable artifact verification.
from check_group_core import checks
from group_core import (MODES, SURFACE, VARIANCE_FLOOR, build_inputs, fit, fit_statistics,
    group_values, pair_design, objective, model_scores, choose_scores, load_model, predict)
from experiment_io import load_split, disagreement, metrics
from fusion_core import grouped_folds

ROOT, PROJECT, OLD_RUN = io.ROOT, io.PROJECT, io.OLD_RUN
FIT_RECORDS = []


def blas_threads():
    found = []
    for path in (Path(np.__file__).parent.parent / 'numpy.libs').glob('*openblas*.dll'):
        library = ctypes.CDLL(str(path))
        function = getattr(library, 'scipy_openblas_get_num_threads64_')
        function.restype = ctypes.c_int
        found.append({'library': path.name, 'threads': function()})
    if os.name == 'nt' and not found:
        raise RuntimeError('Could not verify the known NumPy OpenBLAS thread setting')
    assert all(row['threads'] <= 4 for row in found)
    return found


def cpu_name():
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    return platform.processor()


def export_model(path, model):
    path.mkdir(parents=True)
    np.savez(path / 'model.npz', **{k: model[k] for k in ('weights', 'mean', 'scale')})
    dim = len(model['weights'])
    io.save(path / 'config.json', {'method': 'v6_answer_group_pair', 'schema_version': 1,
        'mode': model['mode'], 'dimension': dim, 'lambda': 0.01, 'intercept': 'none',
        'features': [f'embedding_{i}' for i in range(1024)] + SURFACE +
                    (['support_fraction', 'mean_embedding_squared_deviation'] if dim == 1032 else []),
        'variance_floor': VARIANCE_FLOOR, 'variance_below_floor_scale': 1,
        'fit_split': 'train', 'validation_used_for_fitting': False,
        'question_numbers_max': 16, 'question_characters_max': 20000,
        'question_number_abs_and_denominator_max': 10**12,
        'numeric_surface_origin': 'CaspianW/dase7506-mp2@6709b9ff3cfaea1fed0a4f897f6796358ccfcd02',
        'valid_answers_only': True, 'all_unparseable_fallback': 0,
        'tie_break': 'canonical answer, text, float64 embedding bytes'})


def fit_one(bank, x, parsed, groups, gv, y, fit_rows, mode, stats, folder, run, tag):
    folder.mkdir(parents=True)
    tick = time.perf_counter()
    X, weights, audit = pair_design(x, gv, parsed, groups, y, fit_rows, mode, stats)
    design_seconds = time.perf_counter() - tick
    np.savez_compressed(folder / 'pair_audit.npz', **audit)
    call = {'call': len(FIT_RECORDS) + 1, 'tag': tag, 'mode': mode, 'state': 'started',
            'positive_pair_rows': len(weights), 'equivalent_ordered_pair_rows': len(weights)*2,
            'informative_questions': len(audit['informative_questions']),
            'design_seconds': design_seconds}
    FIT_RECORDS.append(call)
    io.save(run / 'fit_progress.json', FIT_RECORDS)
    tick = time.perf_counter()
    w, record = fit(X, weights, maxiter=2000, gtol=1e-6)
    seconds = time.perf_counter() - tick
    io.save(folder / 'optimization.json', record)
    value, gradient = objective(w, X, weights)
    call.update({'state': 'completed', 'fit_seconds': seconds, 'iterations': record['iterations'],
                 'converged': record['converged'], 'gradient_inf': float(np.abs(gradient).max())})
    io.save(run / 'fit_progress.json', FIT_RECORDS)
    if not record['converged'] or np.abs(gradient).max() > 1e-6:
        raise RuntimeError(f'Nonconverged model: {tag} {mode}')
    assert abs(value - record['final_loss']) < 1e-12
    dim = len(w)
    model = {'mode': mode, 'weights': w, 'mean': stats['mean'][:dim], 'scale': stats['scale'][:dim]}
    export_model(folder / 'checkpoint', model)
    reloaded = load_model(folder / 'checkpoint')
    assert all(np.array_equal(model[k], reloaded[k]) for k in ('mean', 'scale', 'weights'))
    call['checkpoint'] = str((folder / 'checkpoint').relative_to(ROOT))
    call['checkpoint_sha256'] = {p.name: io.sha(p) for p in (folder / 'checkpoint').iterdir()}
    io.save(run / 'fit_progress.json', FIT_RECORDS)
    print(f"{tag} {mode}: iter={record['iterations']} grad={call['gradient_inf']:.3g} seconds={seconds:.3f}", flush=True)
    return model


def compare_records(bank, parsed, y, hard, folds, choices, ce):
    result = []
    b1 = np.asarray(choices['candidate_pair'])
    for mode in MODES:
        selected = np.asarray(choices[mode])
        correct = y[np.arange(len(y)), selected].astype(bool)
        patterns = {}
        for i, values in enumerate(parsed):
            counts = {}
            for v in values:
                if v is not None:
                    counts[v] = counts.get(v, 0) + 1
            key = ','.join(map(str, sorted(counts.values(), reverse=True))) or 'none'
            item = patterns.setdefault(key, {'n': 0, 'correct': 0})
            item['n'] += 1; item['correct'] += int(correct[i])
        result.append({'mode': mode, **metrics(correct, hard),
            'versus_ce': io.comparisons(selected, ce, parsed, y, hard),
            'versus_candidate_pair': io.comparisons(selected, b1, parsed, y, hard),
            'by_fold': [{'fold': f, **metrics(correct[folds == f], hard[folds == f])} for f in range(5)],
            'by_valid_answer_support_pattern': patterns})
    return result


def cross_validate(bank, arrays, x, parsed, groups, gv, y, hard, partition, seed, run):
    folder = run / partition
    folder.mkdir()
    folds = grouped_folds(bank, seed=seed)
    with np.load(OLD_RUN / partition / 'oof_scores.npz', allow_pickle=False) as z:
        assert np.array_equal(folds, z['fold']) and np.array_equal(y, z['labels'])
        ce = z[io.NAME].argmax(1)
    np.savez_compressed(folder / 'folds.npz', folds=folds, hard=hard, labels=y)
    oof = {mode: np.full((len(y), 4), np.nan) for mode in MODES}
    choices = {mode: np.full(len(y), -1, dtype=int) for mode in MODES}
    row_audits = []
    for f in range(5):
        fit_rows, held = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        assert not ({' '.join(bank[i]['question'].split()) for i in fit_rows} &
                    {' '.join(bank[i]['question'].split()) for i in held})
        stats = fit_statistics(x, gv, fit_rows)
        stats_file = folder / f'fold_{f}_statistics.npz'
        np.savez(stats_file, **stats, train_rows=fit_rows, held_rows=held)
        row_audits.append({'fold': f, 'train_questions': len(fit_rows), 'held_questions': len(held),
            'base_scaler_candidate_rows': 4*len(fit_rows),
            'extra_scaler_original_group_rows': sum(len(groups[q]) for q in fit_rows),
            'question_overlap': 0, 'statistics_sha256': io.sha(stats_file)})
        hb = [bank[i] for i in held]
        hp = [parsed[i] for i in held]; hg = [groups[i] for i in held]; hv = [gv[i] for i in held]
        for mode in MODES:
            model = fit_one(bank, x, parsed, groups, gv, y, fit_rows, mode, stats,
                folder / mode / f'fold_{f}', run, f'{partition}/fold{f}')
            values = model_scores(hb, x[held], hp, hg, hv, model)
            selected = choose_scores(hb, x[held], hp, values)
            oof[mode][held] = values
            choices[mode][held] = selected
    assert all(np.isfinite(v).all() for v in oof.values())
    assert all(((v >= 0) & (v < 4)).all() for v in choices.values())
    np.savez_compressed(folder / 'oof_scores.npz', folds=folds, labels=y, hard=hard,
        **{f'scores_{mode}': oof[mode] for mode in MODES},
        **{f'selected_{mode}': choices[mode] for mode in MODES})
    records = compare_records(bank, parsed, y, hard, folds, choices, ce)
    io.save(folder / 'summary.json', {'seed': seed, 'records': records, 'fold_row_audits': row_audits})
    return records


def execute(run):
    start = time.perf_counter()
    status = io.read(ROOT / 'STATUS.json')
    assert status['stage_a'] == 'completed' and status['stage_b'] == 'not_started'
    protected = io.protected_snapshot()
    budget = io.read(ROOT / 'BUDGET.json')
    assert budget['stage_b']['new_fits'] == 33
    threads = blas_threads()
    source = {name: io.sha(ROOT / 'code' / name) for name in
        ('group_core.py', 'check_group_core.py', 'stage_b.py', 'stage_a.py')}
    io.save(run / 'started.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': io.sha(ROOT / 'PLAN.md'), 'budget_sha256': io.sha(ROOT / 'BUDGET.json'),
        'sources_sha256': source, 'protected_before': protected, 'max_real_fits': 33,
        'new_validation_calls': 0, 'new_test_calls': 0, 'native_blas_threads': threads,
        'scaler_rule_interpretation': 'Population variance <1e-8 -> scale one, literal stage-B PLAN wording; shared across controls.',
        'optimizer': 'Frozen NumPy L-BFGS two-loop/Armijo40 adapted to weighted pair objective; memory10, maxiter2000, gtol1e-6'})
    synthetic = checks()
    io.save(run / 'synthetic_checks.json', synthetic)
    bank, arrays, y = load_split('train')
    hard = disagreement(bank)
    assert len(bank) == 4755 and int(hard.sum()) == 2114 and int(y.sum()) == 14353
    tick = time.perf_counter()
    x, parsed, groups = build_inputs(bank, arrays)
    gv = group_values(x, groups)
    feature_seconds = time.perf_counter() - tick
    io.save(run / 'inputs.json', {'questions': len(bank), 'hard': int(hard.sum()), 'positive_labels': int(y.sum()),
        'base_shape': list(x.shape), 'feature_seconds': feature_seconds,
        'train_data_sha256': {p: io.sha(io.DATA / p) for p in
            ('gsm8k_train.jsonl', 'candidates/gsm8k_train.jsonl', 'candidates/gsm8k_train.npz')},
        'original_valid_answer_groups': sum(len(g) for g in groups)})
    primary = cross_validate(bank, arrays, x, parsed, groups, gv, y, hard, 'primary', 20261022, run)
    chosen = max((r for r in primary if r['mode'] != 'candidate_pair'),
        key=lambda r: (r['disagreement']['correct'], r['all']['correct'], r['mode'] == 'group_mean'))
    selection = {'selected_group_mode': chosen['mode'], 'selected_from': ['group_mean', 'group_mean_context'],
        'primary_disagreement_correct': chosen['disagreement']['correct'],
        'primary_overall_correct': chosen['all']['correct'],
        'rule': 'primary disagreement correct, primary overall correct, fewer features group_mean',
        'confirmation_scores_not_decoded': True, 'validation_not_accessed': True, 'test_not_accessed': True,
        'source_sha256': source, 'saved_utc': datetime.now(timezone.utc).isoformat()}
    io.save(run / 'selection_before_confirmation.json', selection)
    selection_hash = io.sha(run / 'selection_before_confirmation.json')
    print('GROUP SELECTION FROZEN: ' + chosen['mode'], flush=True)
    confirmation = cross_validate(bank, arrays, x, parsed, groups, gv, y, hard,
                                   'confirmation', 20261029, run)
    assert io.sha(run / 'selection_before_confirmation.json') == selection_hash
    stats = fit_statistics(x, gv, np.arange(len(bank)))
    np.savez(run / 'full_statistics.npz', **stats, train_rows=np.arange(len(bank)))
    full_records = []
    for mode in MODES:
        folder = run / 'full' / mode
        model = fit_one(bank, x, parsed, groups, gv, y, np.arange(len(bank)), mode,
                         stats, folder, run, 'full_train')
        full_records.append({'mode': mode, 'checkpoint': str((folder / 'checkpoint').relative_to(ROOT)),
                             'checkpoint_sha256': FIT_RECORDS[-1]['checkpoint_sha256']})
    assert len(FIT_RECORDS) == 33 and all(r['converged'] for r in FIT_RECORDS)
    assert {p: io.sha(p) for p in protected} == protected
    assert source == {name: io.sha(ROOT / 'code' / name) for name in source}
    assert not io.DENIED
    selected_confirmation = next(r for r in confirmation if r['mode'] == chosen['mode'])
    result = {'stage': 'B', 'status': 'completed', 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'primary': primary, 'confirmation': confirmation, 'selection': selection,
        'selection_unchanged_after_confirmation': True,
        'selected_group_both_cv_above_ce': chosen['disagreement']['correct'] > 1743 and
            selected_confirmation['disagreement']['correct'] > 1749,
        'full_models': full_records, 'new_real_fits': 33, 'all33_converged': True,
        'max_gradient_inf': max(r['gradient_inf'] for r in FIT_RECORDS),
        'fit_seconds_sum': sum(r['fit_seconds'] for r in FIT_RECORDS),
        'design_seconds_sum': sum(r['design_seconds'] for r in FIT_RECORDS),
        'wall_seconds': time.perf_counter() - start, 'feature_seconds': feature_seconds,
        'synthetic_optimizer_calls_in_registered_run': 1,
        'additional_preflight_synthetic_optimizer_calls': 1,
        'official_validation_calls': 0, 'official_test_calls': 0,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'cpu': cpu_name(),
            'threads': 4, 'native_blas_threads': threads, 'model_workers': 1, 'reference_cpu_measured': False},
        'protected_artifacts_unchanged': len(protected), 'denied_accesses': io.DENIED,
        'bank_accesses': sorted(p for p in io.OPENED if Path(p).name.startswith('gsm8k_')),
        'note': 'CV-only model selection. Holdout inputs/labels never fit fold models/scalers. No unbiased generalization claim.'}
    io.save(run / 'summary.json', result)
    print(json.dumps({'stage_b': 'completed', 'selected_group_mode': chosen['mode'],
        'primary': {r['mode']: r['disagreement']['correct'] for r in primary},
        'confirmation': {r['mode']: r['disagreement']['correct'] for r in confirmation},
        'real_fits': 33, 'all_converged': True}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/group_models/stage_b')
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        raise ValueError('Use a new output directory within V6')
    out.mkdir(parents=True)
    try:
        execute(out)
    except Exception as error:
        io.save(out / 'failure.json', {'error': type(error).__name__, 'message': str(error),
            'actual_real_fit_calls': len(FIT_RECORDS), 'fit_records': FIT_RECORDS,
            'denied_accesses': io.DENIED})
        raise


if __name__ == '__main__':
    main()
