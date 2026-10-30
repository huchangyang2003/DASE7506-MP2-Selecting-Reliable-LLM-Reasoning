"""Audit and reproduce the existing CE baseline without fitting any model."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent
OLD = PROJECT / 'V3_Block_Regularization'
RUN = OLD / 'runs/step1'
CODE = OLD / 'code'
NAME = 'uniform_ce_e0.01_h0.01'
CHECKPOINT = RUN / NAME / 'checkpoint'
sys.path.insert(0, str(CODE))
import numpy as np
from common import PROTOCOL, check_data, parse_answer
from experiment_io import DATA, load_split, disagreement, metrics, snapshot_original
from fusion_core import grouped_folds
from linear_core import build_inputs, fit_scaler, loss_gradient
from submission import candidate_scores, predict

PHASE = 'audit'
OPENED = set()
DENIED = []
CONTRACT_READS = set()


def audit(event, args):
    if event.startswith('socket.connect'):
        raise PermissionError('Stage0 is offline')
    if event != 'open' or not isinstance(args[0], (str, bytes)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    name = path.name.lower()
    forbidden = name.startswith(('gsm8k_test', 'gsm8k_audit', 'test-predictions', 'test-score'))
    if PHASE == 'contract' and (name.startswith('gsm8k_') or name.endswith('.jsonl')):
        forbidden = True
    if forbidden:
        DENIED.append({'phase': PHASE, 'path': str(path)})
        raise PermissionError('Forbidden bank/reference access')
    if PHASE == 'contract':
        CONTRACT_READS.add(str(path))
    if name.startswith('gsm8k_') or name.endswith('.jsonl'):
        OPENED.add((PHASE, str(path)))


sys.addaudithook(audit)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def cpu_name():
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    return platform.processor()


def official(runner, args, output):
    command = [sys.executable, '-B', str(CODE / 'guarded_official.py'), runner, *args]
    tick = time.perf_counter()
    result = subprocess.run(command, cwd=CODE, capture_output=True, text=True, encoding='utf-8',
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    elapsed = time.perf_counter() - tick
    output.with_suffix('.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'{runner} failed: {result.stderr}')
    return {'command': command, 'wall_seconds': elapsed, 'returncode': result.returncode,
            'access': read(output.with_suffix('.access.json'))}


def execute(destination):
    global PHASE
    tick = time.perf_counter()
    save(destination / 'started.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Stage0 only; zero fits; one official validation score; no test',
        'script_sha256': sha(__file__), 'plan_sha256': sha(ROOT / 'PLAN.md'),
        'budget_sha256': sha(ROOT / 'BUDGET.json')})
    protected = read(OLD / 'ARTIFACT_MANIFEST.json')['files']
    before = {}
    for relative, metadata in protected.items():
        before[relative] = sha(OLD / relative)
        assert before[relative] == metadata['sha256'], relative
    plan_manifest = read(ROOT / 'PLAN_MANIFEST.json')
    for name, metadata in plan_manifest['files'].items():
        assert sha(ROOT / name) == metadata['sha256'], name
    originals = snapshot_original()
    training = read(RUN / 'training_complete.json')
    inputs = read(RUN / 'inputs.json')
    assert training['no_validation_or_test_opened'] and not training['denied_accesses']
    for name, expected in training['sources_sha256'].items():
        assert sha(CODE / name) == expected, name
    for name in ('common.py', 'predict.py', 'score_predictions.py'):
        assert sha(CODE / name) == sha(PROJECT / 'DASE7506-MP2-main/code' / name), name
    model_record = next(m for m in training['models'] if m['config'] == NAME)
    for name, expected in model_record['checkpoint_sha256'].items():
        assert sha(CHECKPOINT / name) == expected, name
    config = read(CHECKPOINT / 'config.json')
    assert config['objective'] == 'uniform_ce'
    assert config['embedding_l2'] == config['hand_l2'] == 0.01
    assert not config['clipping'] and not config['uses_validation_for_fit']
    assert config['intercept'] == 'fixed_zero'
    check_data(DATA, ('train',))
    for relative, expected in inputs['train_data_sha256'].items():
        assert sha(DATA / relative) == expected, relative
    bank, arrays, y = load_split('train')
    hard = disagreement(bank)
    assert y.shape == (4755, 4) and int(y.sum()) == 14353 and int(hard.sum()) == 2114
    x, names = build_inputs(bank, arrays)
    assert x.shape == (4755, 4, 1071) and names == config['feature_names'] == inputs['feature_names']
    with np.load(CHECKPOINT / 'model.npz', allow_pickle=False) as bundle:
        state = {name: bundle[name].copy() for name in bundle.files}
    assert set(state) == {'mean', 'scale', 'weights', 'bias'}
    assert all(np.isfinite(a).all() for a in state.values())
    assert all(a.dtype == np.float64 for a in state.values())
    mean, scale = fit_scaler(x)
    assert np.array_equal(mean, state['mean']) and np.array_equal(scale, state['scale'])
    assert float(state['bias']) == 0.0
    theta = np.r_[state['weights'], state['bias']]
    value, gradient = loss_gradient(theta, (x - mean) / scale, y, 'uniform_ce', np.full(1071, 0.01))
    optimization = read(RUN / NAME / 'optimization.json')
    assert abs(value - optimization['final_loss']) < 1e-12
    assert float(np.abs(gradient).max()) <= 1e-6
    assert abs(float(np.abs(gradient).max()) - optimization['gradient_inf']) < 1e-12
    oof = []
    for partition, seed, expected_correct in (('primary', 20261022, 1743), ('confirmation', 20261029, 1749)):
        directory = RUN / partition
        with np.load(directory / 'oof_scores.npz', allow_pickle=False) as z:
            folds, archived_y, archived_hard, scores = [z[k].copy() for k in ('fold', 'labels', 'hard', NAME)]
        assert scores.shape == y.shape and np.isfinite(scores).all()
        assert np.array_equal(archived_y, y) and np.array_equal(archived_hard, hard)
        assert np.array_equal(folds, grouped_folds(bank, seed=seed))
        assigned = read(directory / 'fold_assignments.json')
        assert assigned == [{'id': b['id'], 'fold': int(f)} for b, f in zip(bank, folds)]
        audit_rows = read(directory / 'fold_audit.json')
        coverage = np.zeros(len(bank), dtype=int)
        fold_records = []
        for f in range(5):
            tr, ho = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            recorded = next(row for row in audit_rows if row['fold'] == f)
            assert recorded['train_rows'] == tr.tolist() and recorded['held_rows'] == ho.tolist()
            assert not ({' '.join(bank[i]['question'].split()) for i in tr} &
                        {' '.join(bank[i]['question'].split()) for i in ho})
            coverage[ho] += 1
            opt = read(directory / NAME / f'fold_{f}_optimization.json')
            assert opt['converged'] and opt['gradient_inf'] <= 1e-6
            fold_records.append({'fold': f, 'train_questions': len(tr), 'held_questions': len(ho),
                                 'held_disagreement': int(hard[ho].sum()), 'gradient_inf': opt['gradient_inf']})
        assert np.array_equal(coverage, np.ones(len(bank)))
        correct = y[np.arange(len(y)), scores.argmax(1)].astype(bool)
        measured = metrics(correct, hard)
        assert measured['disagreement']['correct'] == expected_correct
        archived_summary = next(s for s in read(directory / 'summary.json') if s['config'] == NAME)
        assert all(measured[k] == archived_summary[k] for k in measured)
        oof.append({'seed': seed, 'path': str((directory / 'oof_scores.npz').relative_to(PROJECT)),
            'sha256': sha(directory / 'oof_scores.npz'), 'score_key': NAME, 'metrics': measured,
            'fold_records': fold_records, 'held_coverage_exactly_once': True,
            'source_and_fold_audits_verified': True,
            'limitation': 'Original fold weights/scalers were not saved; OOF values were hash-verified, not regenerated.'})
    pre = {'passed': True, 'historical_artifacts_verified': len(protected),
        'source_hashes_exact': True, 'official_parser_and_runners_exact': True,
        'train_data_hashes_exact': True, 'dimension': 1071,
        'full_scaler_arrays_exact': True, 'objective_recomputed': value,
        'gradient_inf_recomputed': float(np.abs(gradient).max()), 'oof': oof,
        'new_fits': 0, 'checkpoint': str(CHECKPOINT.relative_to(PROJECT)),
        'checkpoint_sha256': model_record['checkpoint_sha256']}
    save(destination / 'pre_validation_audit.json', pre)
    print('TRAIN provenance and both OOF audits passed; no refitting.', flush=True)
    PHASE = 'validation'
    pred = destination / 'validation-predictions.json'
    score = destination / 'validation-score.json'
    common_args = ['--split', 'validation', '--data-dir', str(DATA)]
    prediction_run = official('predict.py', ['--method', str(CODE / 'submission.py'),
        '--checkpoint', str(CHECKPOINT), *common_args, '--output', str(pred)], pred)
    scoring_run = official('score_predictions.py', [*common_args, '--predictions', str(pred),
                                                  '--output', str(score)], score)
    prediction, official_score = read(pred), read(score)
    historic = read(RUN / 'validation' / NAME / 'validation-predictions.json')
    assert prediction['protocol'] == PROTOCOL and prediction['split'] == 'validation'
    assert prediction['ids'] == historic['ids'] and prediction['selected'] == historic['selected']
    vb, va, vy = load_split('validation')
    vh = disagreement(vb)
    assert prediction['ids'] == [b['id'] for b in vb]
    selected = np.asarray(prediction['selected'], dtype=int)
    measured = metrics(vy[np.arange(len(vy)), selected].astype(bool), vh)
    assert official_score['correct'] == measured['all']['correct'] == 748
    assert measured['disagreement']['correct'] == 305 and measured['other']['correct'] == 443
    assert measured['all']['n'] == official_score['n'] == 798
    assert measured['disagreement']['n'] == 355
    assert not prediction_run['access']['denied'] and not scoring_run['access']['denied']
    assert all(not (Path(p).parent.name != 'candidates' and Path(p).name == 'gsm8k_validation.jsonl')
               for p in prediction_run['access']['opened'])
    PHASE = 'contract'
    small, small_a = vb[:7], {k: v[:7] for k, v in va.items()}
    base_scores = candidate_scores(small, small_a, CHECKPOINT)
    base_choices = predict(small, small_a, CHECKPOINT)
    assert all(type(c) is int and 0 <= c < 4 for c in base_choices)
    assert predict([], {k: v[:0] for k, v in va.items()}, CHECKPOINT) == []
    for count in (1, 3, 7):
        batch = predict(small[:count], {k: v[:count] for k, v in small_a.items()}, CHECKPOINT)
        assert batch == base_choices[:count]
    changed = copy.deepcopy(small)
    for i, b in enumerate(changed):
        b['id'] = f'stage0-new-id-{i}'
    assert predict(changed, small_a, CHECKPOINT) == base_choices
    permutation = np.array([2, 0, 3, 1])
    for b in changed:
        b['candidates'] = [{**b['candidates'][j], 'index': k} for k, j in enumerate(permutation)]
    perm_a = {k: v[:, permutation] for k, v in small_a.items()}
    perm_scores = candidate_scores(changed, perm_a, CHECKPOINT)
    assert np.allclose(perm_scores, base_scores[:, permutation], rtol=1e-9, atol=1e-9)
    perm_choices = predict(changed, perm_a, CHECKPOINT)
    assert [parse_answer(changed[i]['candidates'][j]['text']) for i, j in enumerate(perm_choices)] == \
           [parse_answer(small[i]['candidates'][j]['text']) for i, j in enumerate(base_choices)]
    expected_reads = {str((CHECKPOINT / n).resolve()) for n in ('config.json', 'model.npz')}
    assert CONTRACT_READS <= expected_reads, sorted(CONTRACT_READS - expected_reads)
    PHASE = 'final_integrity'
    assert snapshot_original() == originals
    assert {relative: sha(OLD / relative) for relative in protected} == before
    for name, expected in model_record['checkpoint_sha256'].items():
        assert sha(CHECKPOINT / name) == expected
    assert not DENIED
    with zipfile.ZipFile(CHECKPOINT / 'model.npz') as z:
        assets = sum(i.file_size for i in z.infolist()) + (CHECKPOINT / 'config.json').stat().st_size
    assert assets == model_record['uncompressed_bytes'] <= 16 * 2**20
    result = {'status': 'passed', 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'stage': '0', 'new_training_fits': 0, 'official_validation_scoring_calls': 1,
        'official_test_calls': 0, 'all798_candidate_indices_identical_to_historical': True,
        'validation': measured, 'oof': oof, 'pre_validation_audit': 'pre_validation_audit.json',
        'prediction_process': prediction_run, 'scoring_process': scoring_run,
        'complete_validation_wall_seconds': prediction_run['wall_seconds'] + scoring_run['wall_seconds'],
        'uncompressed_inference_asset_bytes': assets,
        'interface_checks': {'new_ids': True, 'empty_input': True, 'batch_sizes': [1, 3, 7],
                             'batch_independence': True, 'candidate_score_permutation': True,
                             'candidate_answer_permutation': True, 'input_only_read_audit': True},
        'interface_checkpoint_reads': sorted(CONTRACT_READS),
        'old169_research_artifacts_unchanged': True, 'frozen_model_and_report_unchanged': True,
        'no_test_bank_or_references_read': True, 'denied_accesses': DENIED,
        'parent_bank_accesses': [{'phase': phase, 'path': path} for phase, path in sorted(OPENED)],
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
            'cpu': cpu_name(), 'platform': platform.platform(), 'thread_environment':
            {k: os.environ[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
            'workers': 1, 'reference_cpu_measured': False},
        'stage0_wall_seconds': time.perf_counter() - tick,
        'stage_a_authorized_or_executed': False}
    save(destination / 'summary.json', result)
    print(json.dumps({'status': 'passed', 'validation': measured, 'new_fits': 0,
                      'complete_validation_seconds': result['complete_validation_wall_seconds']}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/baseline/stage0')
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to(ROOT) or destination.exists():
        raise ValueError('Use a fresh output directory inside V6')
    destination.mkdir(parents=True)
    try:
        execute(destination)
    except Exception as error:
        save(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error),
            'phase': PHASE, 'new_fits': 0, 'denied_accesses': DENIED})
        raise


if __name__ == '__main__':
    main()
