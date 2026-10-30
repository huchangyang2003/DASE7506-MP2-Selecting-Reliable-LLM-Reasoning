"""Nonempty subsets within existing numeric-answer groups, with fixed B statistics.

Design inspired by CaspianW/dase7506-mp2 at
6709b9ff3cfaea1fed0a4f897f6796358ccfcd02, train_surface.py.
No new texts, labels, scaler fitting or inference-time subset augmentation.
"""
from itertools import combinations, product
import numpy as np


def subset_values(x, members, mode):
    if mode not in ('group_mean', 'group_mean_context'):
        raise ValueError('Only the already selected group model may be augmented')
    values = []
    for size in range(1, len(members)+1):
        for subset in combinations(members, size):
            block = x[list(subset)]
            mean = block.mean(axis=0)
            value = mean
            if mode == 'group_mean_context':
                spread = np.square(block[:, :1024]-mean[:1024]).mean()
                value = np.r_[mean, size/4.0, spread]
            values.append((value, sum(1 << j for j in subset)))
    assert len(values) == 2**len(members)-1
    return values


def subset_design(x, groups, labels, fit_rows, mode, stats):
    dim = 1032 if mode == 'group_mean_context' else 1030
    differences, weights, owners = [], [], []
    pmasks, nmasks, full_pmasks, full_nmasks = [], [], [], []
    subset_counts = np.zeros(len(x), dtype=int)
    original_pair_counts = np.zeros(len(x), dtype=int)
    for q in fit_rows:
        good, bad = [], []
        for g, (_, members) in enumerate(groups[q]):
            target = np.asarray(labels[q, members], dtype=bool)
            if not np.all(target == target[0]):
                raise ValueError('Equivalent numeric answers need the same label')
            (good if target[0] else bad).append(g)
        if not good or not bad:
            continue
        expanded = [subset_values(x[q], members, mode) for _, members in groups[q]]
        subset_counts[q] = sum(len(v) for v in expanded)
        original_pair_counts[q] = len(good)*len(bad)
        for p in good:
            for n in bad:
                pair_weight = 1.0 / (original_pair_counts[q]*len(expanded[p])*len(expanded[n]))
                full_p = sum(1 << j for j in groups[q][p][1])
                full_n = sum(1 << j for j in groups[q][n][1])
                for (pv, pm), (nv, nm) in product(expanded[p], expanded[n]):
                    differences.append(pv-nv); weights.append(pair_weight); owners.append(q)
                    pmasks.append(pm); nmasks.append(nm)
                    full_pmasks.append(full_p); full_nmasks.append(full_n)
    if not differences:
        raise ValueError('No informative fitting questions')
    X = np.asarray(differences, dtype=np.float64) / stats['scale'][:dim]
    w = np.asarray(weights, dtype=np.float64)
    owners = np.asarray(owners, dtype=int)
    totals = np.bincount(owners, weights=w, minlength=len(x))
    informative = np.flatnonzero(totals > 0)
    assert set(informative) <= set(fit_rows)
    assert np.allclose(totals[informative], 1.0, rtol=1e-13, atol=1e-13)
    return X, w, {'owners': owners, 'positive_mask': np.asarray(pmasks, dtype=np.int8),
        'negative_mask': np.asarray(nmasks, dtype=np.int8),
        'full_positive_mask': np.asarray(full_pmasks, dtype=np.int8),
        'full_negative_mask': np.asarray(full_nmasks, dtype=np.int8),
        'row_weights': w, 'question_total_weights': totals, 'informative_questions': informative,
        'subset_counts_per_question': subset_counts, 'original_group_pair_counts': original_pair_counts}
