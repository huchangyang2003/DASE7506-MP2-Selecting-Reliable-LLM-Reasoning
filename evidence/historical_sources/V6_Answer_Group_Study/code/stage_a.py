"""Apply only the four preplanned answer-pooling strengths to frozen TRAIN OOF."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import numpy as np
from pooling_core import ALPHAS, answers, choose
from check_pooling import checks
from experiment_io import DATA, load_split, disagreement, metrics
from fusion_core import grouped_folds

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent
OLD = PROJECT / 'V3_Block_Regularization'
OLD_RUN = OLD / 'runs/step1'
NAME = 'uniform_ce_e0.01_h0.01'
OPENED, DENIED = set(), []


def access_guard(event, args):
    if event.startswith('socket.connect'):
        raise PermissionError('StageA is offline')
    if event != 'open' or not isinstance(args[0], (str, bytes)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    lower = path.as_posix().lower()
    name = path.name.lower()
    forbidden = name.startswith(('gsm8k_validation', 'gsm8k_test', 'gsm8k_audit',
                                 'validation-predictions', 'validation-score',
                                 'test-predictions', 'test-score')) or '/validation/' in lower
    if forbidden:
        DENIED.append(str(path))
        raise PermissionError('Only TRAIN is permitted in stageA')
    OPENED.add(str(path))


sys.addaudithook(access_guard)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def may_hash(relative):
    lower = relative.lower().replace('\\', '/')
    return not any(fragment in lower for fragment in ('validation', 'gsm8k_test', 'gsm8k_audit',
                                                     'test-predictions', 'test-score'))


def protected_snapshot():
    result = {}
    for base, manifest in ((OLD, OLD / 'ARTIFACT_MANIFEST.json'),
                           (ROOT, ROOT / 'ARTIFACT_MANIFEST.json')):
        for relative, entry in read(manifest)['files'].items():
            if may_hash(relative):
                path = base / relative
                digest = sha(path)
                assert digest == entry['sha256'], str(path)
                result[str(path)] = digest
    frozen = PROJECT / 'Final_Optimized_Solution'
    for relative, entry in read(frozen / 'FREEZE.json')['files'].items():
        path = frozen / relative
        digest = sha(path)
        assert digest == entry['sha256']
        result[str(path)] = digest
    path = frozen / 'output/pdf/DASE7506_MP2_Final_Report.pdf'
    result[str(path)] = sha(path)
    return result


def comparisons(selected, baseline, parsed, labels, hard):
    index = np.arange(len(labels))
    new = labels[index, selected].astype(bool)
    old = labels[index, baseline].astype(bool)
    different = selected != baseline
    both_valid = np.asarray([parsed[i][a] is not None and parsed[i][b] is not None
                             for i, (a, b) in enumerate(zip(selected, baseline))])
    same_valid_answer = np.asarray([parsed[i][a] is not None and parsed[i][a] == parsed[i][b]
                                    for i, (a, b) in enumerate(zip(selected, baseline))])
    result = {}
    for name, mask in (('all', np.ones(len(labels), dtype=bool)), ('disagreement', hard), ('other', ~hard)):
        result[name] = {'fixed': int((mask & new & ~old).sum()),
            'broken': int((mask & old & ~new).sum()),
            'net_correct': int((mask & new).sum() - (mask & old).sum()),
            'different_indices': int((mask & different).sum()),
            'different_indices_same_valid_answer': int((mask & different & same_valid_answer).sum()),
            'different_valid_answers': int((mask & different & both_valid & ~same_valid_answer).sum()),
            'different_indices_with_unparseable': int((mask & different & ~both_valid).sum())}
    result['invalid_selected'] = int(sum(parsed[i][j] is None for i, j in enumerate(selected)))
    result['new_invalid_selected'] = int(sum(parsed[i][j] is None and parsed[i][baseline[i]] is not None
                                            for i, j in enumerate(selected)))
    return result


def evaluate_partition(bank, arrays, y, hard, partition, seed, expected, out):
    old = OLD_RUN / partition
    with np.load(old / 'oof_scores.npz', allow_pickle=False) as saved:
        folds, logits = saved['fold'].copy(), saved[NAME].copy()
        assert np.array_equal(saved['labels'], y) and np.array_equal(saved['hard'], hard)
    assert logits.shape == (4755, 4) and np.isfinite(logits).all()
    assert np.array_equal(folds, grouped_folds(bank, seed=seed))
    base = logits.argmax(1)
    assert int(y[np.arange(len(y)), base][hard].sum()) == expected
    parsed = answers(bank)
    rows, all_pooled, all_selected = [], [], []
    out.mkdir(parents=True)
    # Primary order is fixed; the confirmation data are opened only after selection is saved.
    for alpha in ALPHAS:
        selected, pooled, ties = choose(bank, logits, alpha, arrays['embeddings'])
        selected = np.asarray(selected, dtype=int)
        correct = y[np.arange(len(y)), selected].astype(bool)
        if alpha == 0:
            assert np.array_equal(selected, base)
        by_pattern = {}
        for i, values in enumerate(parsed):
            counts = {}
            for value in values:
                if value is not None:
                    counts[value] = counts.get(value, 0) + 1
            pattern = ','.join(map(str, sorted(counts.values(), reverse=True))) or 'none'
            item = by_pattern.setdefault(pattern, {'n': 0, 'correct': 0})
            item['n'] += 1
            item['correct'] += int(correct[i])
        rows.append({'alpha': alpha, **metrics(correct, hard),
            'versus_ce': comparisons(selected, base, parsed, y, hard),
            'by_fold': [{'fold': f, **metrics(correct[folds == f], hard[folds == f])} for f in range(5)],
            'by_valid_answer_support_pattern': by_pattern,
            'top_score_exact_ties': len(ties),
            'top_score_ties_between_different_answers': int(sum(
                len({parsed[i][j] for j in np.flatnonzero(pooled[i] == pooled[i].max())}) > 1 for i in ties))})
        all_pooled.append(pooled)
        all_selected.append(selected)
    np.savez_compressed(out / 'pooled_oof.npz', alpha=np.asarray(ALPHAS),
        pooled_scores=np.stack(all_pooled), selected=np.stack(all_selected),
        folds=folds, labels=y, hard=hard)
    save(out / 'summary.json', {'partition': partition, 'seed': seed,
        'source_oof_sha256': sha(old / 'oof_scores.npz'), 'records': rows})
    return rows, logits


def execute(out):
    start = time.perf_counter()
    assert read(ROOT / 'STATUS.json')['stage0'] == 'passed'
    assert read(ROOT / 'STATUS.json')['stage_a'] == 'not_started'
    protected = protected_snapshot()
    budget = read(ROOT / 'BUDGET.json')
    assert tuple(budget['stage_a']['alphas']) == ALPHAS
    for name, metadata in read(ROOT / 'PLAN_MANIFEST.json')['files'].items():
        assert sha(ROOT / name) == metadata['sha256']
    synthetic = checks()
    save(out / 'synthetic_checks.json', synthetic)
    save(out / 'started.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': sha(ROOT / 'PLAN.md'), 'budget_sha256': sha(ROOT / 'BUDGET.json'),
        'source_sha256': {name: sha(ROOT / 'code' / name) for name in
            ('stage_a.py', 'pooling_core.py', 'pooling_submission.py', 'check_pooling.py')},
        'protected_before': protected, 'alphas': ALPHAS, 'fits': 0,
        'validation_calls': 0, 'test_calls': 0})
    bank, arrays, y = load_split('train')
    hard = disagreement(bank)
    assert len(bank) == 4755 and int(hard.sum()) == 2114
    primary, logits = evaluate_partition(bank, arrays, y, hard, 'primary', 20261022, 1743, out / 'primary')
    selected = max(primary, key=lambda r: (r['disagreement']['correct'], r['all']['correct'], -r['alpha']))
    selection = {'selected_alpha': selected['alpha'], 'selection_split': 'primary TRAIN OOF only',
        'rule': 'disagreement correct; overall correct; smaller alpha',
        'primary_correct': selected['all']['correct'],
        'primary_disagreement_correct': selected['disagreement']['correct'],
        'confirmation_not_yet_opened': True, 'validation_not_opened': True,
        'test_not_opened': True, 'selected_at_utc': datetime.now(timezone.utc).isoformat()}
    save(out / 'selection_before_confirmation.json', selection)
    selection_hash = sha(out / 'selection_before_confirmation.json')
    print('Primary selection frozen: alpha=' + str(selected['alpha']), flush=True)
    confirmation, _ = evaluate_partition(bank, arrays, y, hard, 'confirmation', 20261029, 1749, out / 'confirmation')
    assert sha(out / 'selection_before_confirmation.json') == selection_hash
    confirm = next(row for row in confirmation if row['alpha'] == selected['alpha'])
    # Check input-only permutation and batch behavior on actual train inputs; no new fit.
    n = 32
    small, emb, z = bank[:n], arrays['embeddings'][:n], logits[:n]
    perm = np.array([2, 0, 3, 1])
    changed = deepcopy(small)
    for i, row in enumerate(changed):
        row['id'] = f'pooling-a-new-id-{i}'
        row['candidates'] = [{**row['candidates'][j], 'index': k} for k, j in enumerate(perm)]
    contracts = []
    for alpha in ALPHAS:
        picks, pooled, _ = choose(small, z, alpha, emb)
        perm_picks, perm_pooled, _ = choose(changed, z[:, perm], alpha, emb[:, perm])
        assert np.array_equal(perm_pooled, pooled[:, perm])
        a, b = answers(small), answers(changed)
        assert [a[i][j] for i, j in enumerate(picks)] == [b[i][j] for i, j in enumerate(perm_picks)]
        for count in (1, 3, 7, 32):
            assert choose(small[:count], z[:count], alpha, emb[:count])[0] == picks[:count]
        contracts.append({'alpha': alpha, 'real_train_rows': n, 'permutation_scores_exact': True,
            'permutation_answers_exact': True, 'new_ids': True, 'batch_sizes': [1, 3, 7, 32]})
    checkpoint = None
    asset_bytes = None
    if selected['alpha'] != 0:
        checkpoint = out / 'selected_checkpoint'
        baseline = checkpoint / 'baseline'
        baseline.mkdir(parents=True)
        original_cp = OLD_RUN / NAME / 'checkpoint'
        for name in ('model.npz', 'config.json'):
            shutil.copyfile(original_cp / name, baseline / name)
            assert sha(original_cp / name) == sha(baseline / name)
        save(checkpoint / 'config.json', {'method': 'v6_ce_answer_pooling', 'alpha': selected['alpha'],
            'temperature': 1, 'source_checkpoint': str(original_cp.relative_to(PROJECT)),
            'source_oof': 'primary', 'selection_sha256': selection_hash,
            'feature_source': 'unchanged V3_Block_Regularization/code'})
        with zipfile.ZipFile(baseline / 'model.npz') as zf:
            asset_bytes = sum(i.file_size for i in zf.infolist())
        asset_bytes += (baseline / 'config.json').stat().st_size + (checkpoint / 'config.json').stat().st_size
        assert asset_bytes < 16 * 2**20
    assert {path: sha(path) for path in protected} == protected
    assert not DENIED
    result = {'stage': 'A', 'status': 'completed', 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'primary': primary, 'confirmation': confirmation, 'selection': selection,
        'selected_confirmation_disagreement_correct': confirm['disagreement']['correct'],
        'both_cv_strictly_above_ce': selected['disagreement']['correct'] > 1743 and confirm['disagreement']['correct'] > 1749,
        'new_validation_candidate': selected['alpha'] != 0,
        'validation_candidate_alpha': selected['alpha'] if selected['alpha'] != 0 else None,
        'selected_checkpoint': str(checkpoint.relative_to(ROOT)) if checkpoint else None,
        'uncompressed_asset_bytes': asset_bytes, 'selection_unchanged_after_confirmation': True,
        'new_fits': 0, 'official_validation_scoring_calls': 0, 'official_test_calls': 0,
        'actual_train_contracts': contracts, 'synthetic_checks': synthetic,
        'protected_artifacts_verified_unchanged': len(protected),
        'data_accesses': sorted(p for p in OPENED if Path(p).name.startswith('gsm8k_')),
        'denied_accesses': DENIED, 'wall_seconds': time.perf_counter() - start,
        'note': 'OOF postprocessing only. No inference latency measurement, new validation, fit, or claim of unbiased CV.'}
    save(out / 'summary.json', result)
    print(json.dumps({'selected_alpha': selected['alpha'],
        'primary': [{'a': r['alpha'], 'hard_correct': r['disagreement']['correct']} for r in primary],
        'confirmation': [{'a': r['alpha'], 'hard_correct': r['disagreement']['correct']} for r in confirmation],
        'new_validation_candidate': result['new_validation_candidate'], 'new_fits': 0}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/pooling/stage_a')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Use a fresh output directory within V6')
    output.mkdir(parents=True)
    try:
        execute(output)
    except Exception as error:
        save(output / 'failure.json', {'type': type(error).__name__, 'message': str(error),
             'new_fits': 0, 'denied_accesses': DENIED})
        raise


if __name__ == '__main__':
    main()
