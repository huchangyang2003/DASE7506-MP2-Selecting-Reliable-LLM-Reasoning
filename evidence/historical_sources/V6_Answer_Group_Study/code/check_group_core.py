"""Stage-B synthetic gradient, grouping, normalization and isolation checks."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
from copy import deepcopy
from fractions import Fraction
import json
import numpy as np
from group_core import (MODES, build_inputs, choose_scores, fit, fit_statistics,
                        group_values, model_scores, objective, pair_design, question_numbers)


def checks():
    rng = np.random.default_rng(20261030)
    values = [['1/2', '0.5', '2', None], ['9', '9', '8', '7'], ['5']*4, ['1', '2', '3', '4']]
    bank = [{'id': str(i), 'question': 'Given 1,000 and 1/2 and 3 and 0.',
             'candidates': [{'index': j, 'text': f'c{j} #### {a}' if a is not None else 'unknown'}
                            for j, a in enumerate(row)]} for i, row in enumerate(values)]
    arrays = {'embeddings': rng.normal(size=(4, 4, 1024))}
    labels = np.asarray([[1, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1], [0, 0, 0, 1]], dtype=float)
    x, parsed, groups = build_inputs(bank, arrays)
    gv = group_values(x, groups)
    stats = fit_statistics(x, gv, [0, 1, 2])
    assert parsed[0][:2] == ['1/2', '1/2']
    assert question_numbers('1,000 1/2 3/0 ' + '9'*81) == [Fraction(1000), Fraction(1, 2)]
    changed = x.copy(); changed[3] *= 100
    changed_stats = fit_statistics(changed, group_values(changed, groups), [0, 1, 2])
    assert all(np.array_equal(stats[k], changed_stats[k]) for k in stats)
    assert np.array_equal(stats['mean'][:1030], x[:3].reshape(-1, 1030).mean(axis=0))
    designs = {}
    for mode in MODES:
        X, weights, audit = pair_design(x, gv, parsed, groups, labels, [0, 1, 2], mode, stats)
        assert np.allclose(audit['question_total_weights'][:3], [1, 1, 0])
        assert set(audit['owners']) == {0, 1}
        assert np.all(audit['positive_mask'] & audit['negative_mask'] == 0)
        assert np.allclose(X[:, 1028], 0)  # Question number count is common to the whole question.
        designs[mode] = (X, weights)
    A, aw = designs['candidate_pair']; B, bw = designs['group_mean']
    assert np.allclose((A*aw[:, None]).sum(0)/aw.sum(), (B*bw[:, None]).sum(0)/bw.sum(), atol=1e-13)
    # Jensen: grouping the candidate differences cannot increase convex loss at fixed weights.
    probe = rng.normal(size=1030) / 100
    assert objective(probe, B, bw)[0] <= objective(probe, A, aw)[0] + 1e-12
    X = rng.normal(size=(13, 7)); weights = rng.uniform(.1, 1, size=13); w = rng.normal(size=7)
    loss, gradient = objective(w, X, weights)
    errors = []
    for j in range(7):
        step = np.zeros(7); step[j] = 1e-6
        numeric = (objective(w+step, X, weights)[0]-objective(w-step, X, weights)[0])/(2e-6)
        errors.append(abs(numeric-gradient[j]))
    assert max(errors) < 1e-7
    solution, record = fit(X, weights)
    assert record['converged'] and np.abs(objective(solution, X, weights)[1]).max() <= 1e-6
    # Duplicate weighted rows preserve the same objective and regularization.
    duplicated = objective(w, np.repeat(X, 2, axis=0), np.repeat(weights/2, 2))
    assert np.isclose(loss, duplicated[0]) and np.allclose(gradient, duplicated[1])
    perm = np.asarray([2, 0, 3, 1]); reordered = deepcopy(bank)
    for i, row in enumerate(reordered):
        row['id'] = f'unseen-{i}'
        row['candidates'] = [{**row['candidates'][j], 'index': k} for k, j in enumerate(perm)]
    rx, rp, rg = build_inputs(reordered, {'embeddings': arrays['embeddings'][:, perm]})
    rgv = group_values(rx, rg)
    for mode in MODES:
        dim = 1032 if mode == 'group_mean_context' else 1030
        model = {'mode': mode, 'mean': stats['mean'][:dim], 'scale': stats['scale'][:dim],
                 'weights': rng.normal(size=dim)/100}
        s = model_scores(bank, x, parsed, groups, gv, model)
        rs = model_scores(reordered, rx, rp, rg, rgv, model)
        assert np.allclose(rs, s[:, perm], rtol=1e-12, atol=1e-12)
        picks = choose_scores(bank, x, parsed, s)
        rpick = choose_scores(reordered, rx, rp, rs)
        assert [parsed[i][j] for i, j in enumerate(picks)] == [rp[i][j] for i, j in enumerate(rpick)]
    invalid_bank = deepcopy(bank[:1])
    for c in invalid_bank[0]['candidates']:c['text'] = 'no answer'
    ix, ip, ig = build_inputs(invalid_bank, {'embeddings': arrays['embeddings'][:1]})
    assert choose_scores(invalid_bank, ix, ip, np.zeros((1, 4))) == [0]
    ex, ep, eg = build_inputs([], {'embeddings': np.empty((0, 4, 1024))})
    assert choose_scores([], ex, ep, np.empty((0, 4))) == []
    return {'passed': True, 'synthetic_optimizer_calls': 1, 'real_data_fits': 0,
        'gradient_finite_difference_max_error': max(errors), 'checks': [
        'canonical equivalent answers', 'bounded question extraction', 'fold-statistic isolation',
        'all four candidates in scaler rows', 'one total pair weight per informative question',
        'invalid and all-positive groups excluded from supervised pairs', 'disjoint positive/negative masks',
        'candidate/group initial gradients match', 'Jensen loss check', 'finite-difference objective gradient',
        'weighted duplication invariant objective', 'fixed-LBFGS gradient stopping',
        'permutation/new-ID answer consistency', 'empty input and all-invalid fallback']}


if __name__ == '__main__':
    print(json.dumps(checks(), indent=2))
