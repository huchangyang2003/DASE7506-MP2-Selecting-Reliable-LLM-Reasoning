"""Reconstruct saved stage-B scalers/pairs/OOF and check deployment without refitting."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
from copy import deepcopy
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
import stage_b as sb
from group_core import (MODES, build_inputs, group_values, load_model, model_scores,
                        choose_scores, predict)
from experiment_io import load_split

CONTRACT = False
CONTRACT_ALLOWED, CONTRACT_READS = set(), set()


def inference_guard(event, args):
    if not CONTRACT or event != 'open' or not isinstance(args[0], (str, bytes)):
        return
    path = str(Path(os.fsdecode(args[0])).resolve())
    CONTRACT_READS.add(path)
    if path not in CONTRACT_ALLOWED:
        raise PermissionError('Deployment may read only the selected checkpoint')


sys.addaudithook(inference_guard)


def decode(mask):
    return [i for i in range(4) if int(mask) & (1 << i)]


def independent_feature(x, bank, q, members, mode):
    members = sorted(members, key=lambda j: (bank[q]['candidates'][j]['text'], x[q, j, :1024].tobytes()))
    if mode == 'candidate_pair':
        assert len(members) == 1
        return x[q, members[0]]
    value = x[q, members].mean(axis=0)
    if mode == 'group_mean_context':
        spread = np.square(x[q, members, :1024] - value[:1024]).mean()
        value = np.r_[value, len(members)/4., spread]
    return value


def audit_fit(folder, mode, bank, x, parsed, groups, y, stats, fit_rows):
    model = load_model(folder / 'checkpoint')
    dim = len(model['weights'])
    assert np.array_equal(model['mean'], stats['mean'][:dim])
    assert np.array_equal(model['scale'], stats['scale'][:dim])
    with np.load(folder / 'pair_audit.npz', allow_pickle=False) as saved:
        audit = {k: saved[k].copy() for k in saved.files}
    expected_informative = [q for q in fit_rows if
        any(y[q, members[0]] for _, members in groups[q]) and
        any(not y[q, members[0]] for _, members in groups[q])]
    assert np.array_equal(audit['informative_questions'], expected_informative)
    totals = np.bincount(audit['owners'], weights=audit['row_weights'], minlength=len(bank))
    assert np.allclose(totals[expected_informative], 1, atol=1e-13, rtol=1e-13)
    assert not set(audit['owners']) - set(fit_rows)
    X, expected_weights = [], []
    for q, pm, nm in zip(audit['owners'], audit['positive_mask'], audit['negative_mask']):
        p, n = decode(pm), decode(nm)
        assert not (int(pm) & int(nm)) and y[q, p].all() and not y[q, n].any()
        assert len({parsed[q][j] for j in p}) == len({parsed[q][j] for j in n}) == 1
        assert all(parsed[q][j] is not None for j in p+n)
        good = [m for _, m in groups[q] if y[q, m[0]]]
        bad = [m for _, m in groups[q] if not y[q, m[0]]]
        expected = 1/(len(good)*len(bad))
        if mode == 'candidate_pair':
            ps = sum(a == parsed[q][p[0]] for a in parsed[q])
            ns = sum(a == parsed[q][n[0]] for a in parsed[q])
            expected /= ps*ns
        else:
            assert set(p) == set(next(m for a, m in groups[q] if a == parsed[q][p[0]]))
            assert set(n) == set(next(m for a, m in groups[q] if a == parsed[q][n[0]]))
        expected_weights.append(expected)
        X.append((independent_feature(x, bank, q, p, mode) -
                  independent_feature(x, bank, q, n, mode)) / model['scale'])
    assert np.allclose(expected_weights, audit['row_weights'], rtol=1e-14, atol=1e-14)
    X = np.asarray(X)
    normalized = np.asarray(expected_weights) / sum(expected_weights)
    margin = X @ model['weights']
    value = float(normalized @ np.logaddexp(0, -margin) + .005*(model['weights']@model['weights']))
    gradient = -(X.T @ (normalized*np.exp(-np.logaddexp(0, margin)))) + .01*model['weights']
    record = sb.io.read(folder / 'optimization.json')
    assert abs(value-record['final_loss']) < 1e-12
    assert float(np.abs(gradient).max()) <= 1e-6
    assert abs(float(np.abs(gradient).max())-record['gradient_inf']) < 1e-12
    return model, {'folder': str(folder.relative_to(sb.ROOT)), 'rows': len(X),
        'informative_questions': len(expected_informative), 'objective': value,
        'gradient_inf': float(np.abs(gradient).max()), 'all_pair_labels_weights_and_owners_verified': True}


def main():
    global CONTRACT, CONTRACT_ALLOWED
    run = sb.ROOT / 'runs/group_models/stage_b'
    output = run / 'audit.json'
    if output.exists():
        raise FileExistsError(output)
    tick = time.perf_counter()
    summary = sb.io.read(run / 'summary.json')
    started = sb.io.read(run / 'started.json')
    assert summary['status'] == 'completed' and summary['new_real_fits'] == 33
    assert {path: sb.io.sha(path) for path in started['protected_before']} == started['protected_before']
    assert {name: sb.io.sha(sb.ROOT/'code'/name) for name in started['sources_sha256']} == started['sources_sha256']
    bank, arrays, y = load_split('train')
    x, parsed, groups = build_inputs(bank, arrays)
    gv = group_values(x, groups)
    reconstructed = []
    full_models = {}
    for partition in ('primary', 'confirmation', 'full'):
        if partition == 'full':
            iterations = [None]
        else:
            with np.load(run / partition / 'oof_scores.npz', allow_pickle=False) as saved:
                archived = {k: saved[k].copy() for k in saved.files}
            iterations = range(5)
        for f in iterations:
            statpath = run / 'full_statistics.npz' if f is None else run / partition / f'fold_{f}_statistics.npz'
            with np.load(statpath, allow_pickle=False) as saved:
                stats = {k: saved[k].copy() for k in saved.files}
            fit_rows = stats['train_rows']
            expected_fit = np.arange(len(bank)) if f is None else np.flatnonzero(archived['folds'] != f)
            assert np.array_equal(fit_rows, expected_fit)
            if f is not None:
                held = np.flatnonzero(archived['folds'] == f)
                assert np.array_equal(stats['held_rows'], held)
            # Recompute the two statistic blocks directly, without the fitting helper.
            base = x[fit_rows].reshape(-1, 1030)
            extras = np.asarray([v[-2:] for q in fit_rows for v in gv[q]])
            mean = np.r_[base.mean(0), extras.mean(0)]
            variance = np.r_[base.var(0), extras.var(0)]
            scale = np.sqrt(variance); scale[variance < 1e-8] = 1
            assert np.array_equal(mean, stats['mean']) and np.array_equal(scale, stats['scale'])
            for mode in MODES:
                folder = run / 'full' / mode if f is None else run / partition / mode / f'fold_{f}'
                model, record = audit_fit(folder, mode, bank, x, parsed, groups, y, stats, fit_rows)
                if f is None:
                    full_models[mode] = model
                else:
                    hb = [bank[i] for i in held]; hp = [parsed[i] for i in held]
                    hg = [groups[i] for i in held]; hv = [gv[i] for i in held]
                    scores = model_scores(hb, x[held], hp, hg, hv, model)
                    assert np.array_equal(scores, archived['scores_'+mode][held])
                    assert np.array_equal(choose_scores(hb, x[held], hp, scores), archived['selected_'+mode][held])
                    record['saved_oof_scores_and_choices_exact'] = True
                reconstructed.append(record)
    contracts, assets = [], {}
    small = bank[:32]; small_a = {k:v[:32] for k,v in arrays.items()}
    for mode in MODES:
        cp = run / 'full' / mode / 'checkpoint'
        CONTRACT_ALLOWED = {str(p.resolve()) for p in cp.iterdir()}
        CONTRACT = True
        choices = predict(small, small_a, cp)
        assert all(type(c) is int and 0 <= c < 4 for c in choices)
        for count in (1, 3, 7, 32):
            assert predict(small[:count], {k:v[:count] for k,v in small_a.items()}, cp) == choices[:count]
        assert predict([], {k:v[:0] for k,v in small_a.items()}, cp) == []
        changed = deepcopy(small)
        for i,row in enumerate(changed):row['id'] = f'b-new-{i}'
        assert predict(changed, small_a, cp) == choices
        perm = np.array([2, 0, 3, 1])
        for row in changed:
            row['candidates'] = [{**row['candidates'][j], 'index':k} for k,j in enumerate(perm)]
        permuted = predict(changed, {k:v[:,perm] for k,v in small_a.items()}, cp)
        pp = [[sb.io.answers([row])[0][j] for j in range(4)] for row in changed]
        assert [parsed[i][j] for i,j in enumerate(choices)] == [pp[i][j] for i,j in enumerate(permuted)]
        unknown = deepcopy(small[:1])
        for row in unknown:
            for c in row['candidates']:c['text'] = 'unknown answer'
        assert predict(unknown, {k:v[:1] for k,v in small_a.items()}, cp) == [0]
        CONTRACT = False
        with zipfile.ZipFile(cp/'model.npz') as z:
            assets[mode] = sum(i.file_size for i in z.infolist()) + (cp/'config.json').stat().st_size
        assert assets[mode] <= 16*2**20
        contracts.append({'mode':mode, 'train_questions':32, 'batch_sizes':[1,3,7,32],
            'new_ids':True, 'permutation_answer_consistency':True, 'empty':True,
            'all_unparseable_fallback':True, 'checkpoint_only_reads':True})
    assert len(reconstructed) == 33 and not sb.io.DENIED
    assert {path: sb.io.sha(path) for path in started['protected_before']} == started['protected_before']
    assert sb.io.read(run/'selection_before_confirmation.json') == summary['selection']
    sb.io.save(output, {'passed':True, 'reconstructed_fits':reconstructed, 'statistics_sets_recomputed':11,
        'exact_oof_models_reproduced':30, 'inference_contracts':contracts,
        'uncompressed_inference_assets':assets, 'inference_reads':sorted(CONTRACT_READS),
        'new_optimizer_calls':0, 'validation_calls':0, 'test_calls':0,
        'native_blas_threads':sb.blas_threads(), 'seconds':time.perf_counter()-tick,
        'bank_accesses':sorted(p for p in sb.io.OPENED if Path(p).name.startswith('gsm8k_'))})
    print({'stage_b_audit':'passed', 'fits_reconstructed':33, 'stats_recomputed':11,
           'oof_models_exact':30, 'assets':assets})


if __name__ == '__main__':
    main()
