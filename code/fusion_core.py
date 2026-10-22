"""Input-only, permutation-equivariant fusion features and grouped folds."""
import numpy as np
from common import parse_answer

FEATURE_NAMES = ['mlp_centered_logit', 'embedding_centered_logit', 'support_centered',
                 'same_answer_mean_mlp_centered', 'same_answer_mean_embedding_centered',
                 'mlp_times_support', 'embedding_times_support', 'valid_centered']


def answer_statistics(bank):
    answers = [[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    same = np.zeros((len(bank), 4, 4), dtype=float)
    valid = np.zeros((len(bank), 4), dtype=float)
    for q, group in enumerate(answers):
        for i, a in enumerate(group):
            valid[q, i] = a is not None
            for j, b in enumerate(group):
                same[q, i, j] = a is not None and a == b
    support = same.sum(axis=2) / 4
    return support, valid, same


def centered(values):
    return values - values.mean(axis=1, keepdims=True)


def meta_features(bank, mlp_scores, embedding_scores):
    a, b = np.asarray(mlp_scores, float), np.asarray(embedding_scores, float)
    if a.shape != (len(bank), 4) or b.shape != a.shape or not np.isfinite([a, b]).all():
        raise ValueError('Invalid base scores')
    a, b = centered(np.clip(a, -30, 30)), centered(np.clip(b, -30, 30))
    support, valid, same = answer_statistics(bank)
    # Invalid parsed answers have no answer group; their group average is zero.
    denominator = np.maximum(same.sum(axis=2), 1)
    group_a = centered((same @ a[..., None])[..., 0] / denominator)
    group_b = centered((same @ b[..., None])[..., 0] / denominator)
    return np.stack([a, b, centered(support), group_a, group_b,
                     centered(a * support), centered(b * support), centered(valid)], axis=2)


def fixed_average(bank, mlp_scores, embedding_scores):
    support, _, _ = answer_statistics(bank)
    sigmoid = lambda z: np.exp(-np.logaddexp(0, -np.asarray(z, float)))
    return (sigmoid(mlp_scores) + sigmoid(embedding_scores) + support) / 3


def grouped_folds(bank, seed=20261022, count=5):
    groups = {}
    for i, row in enumerate(bank):
        key = ' '.join(row['question'].split())
        groups.setdefault(key, []).append(i)
    buckets = {False: [], True: []}
    for key, ids in groups.items():
        hard = any(len({a for c in bank[i]['candidates']
                        if (a := parse_answer(c['text'])) is not None}) >= 2 for i in ids)
        buckets[hard].append(key)
    rng = np.random.default_rng(seed)
    folds = np.full(len(bank), -1, dtype=int)
    for keys in buckets.values():
        for position, shuffled in enumerate(rng.permutation(len(keys))):
            folds[groups[keys[shuffled]]] = position % count
    if (folds < 0).any() or len(set(folds)) != count:
        raise ValueError('Incomplete grouped folds')
    return folds


def fit_standardization(features, indices):
    flat = features[indices].reshape(-1, features.shape[-1])
    mean, scale = flat.mean(axis=0), flat.std(axis=0)
    scale[scale < 1e-8] = 1
    return mean, scale


def mlp_logits(features, mean, scale, params):
    x = np.clip((features - mean) / scale, -8, 8).astype(np.float32)
    return np.maximum(x @ params['W1'] + params['b1'], 0) @ params['W2'] + params['b2'].item()
