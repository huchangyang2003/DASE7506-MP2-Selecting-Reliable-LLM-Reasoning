"""Fixed CE-softmax answer pooling; only current bank inputs and logits are used."""
import numpy as np

from common import parse_answer

ALPHAS = (0.0, 0.25, 0.5, 1.0)


def answers(bank):
    if any(len(row['candidates']) != 4 for row in bank):
        raise ValueError('Exactly four candidates are required')
    return [[parse_answer(c['text']) for c in row['candidates']] for row in bank]


def pool(bank, logits, alpha):
    if alpha not in ALPHAS:
        raise ValueError('Only the four planned strengths are permitted')
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(bank), 4) or not np.isfinite(logits).all():
        raise ValueError('Expected finite [N,4] logits')
    parsed = answers(bank)
    if not bank:
        return np.empty((0, 4), dtype=np.float64), []
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    # Sorted sums give identical denominators after a candidate permutation.
    p = e / np.sort(e, axis=1).sum(axis=1, keepdims=True)
    result = p.copy()
    if alpha != 0:
        for qi, row in enumerate(parsed):
            groups = {}
            for j, value in enumerate(row):
                if value is not None:
                    groups.setdefault(value, []).append(j)
            for members in groups.values():
                mean = np.sort(p[qi, members]).mean()
                result[qi, members] = (1 - alpha) * p[qi, members] + alpha * mean
    return result, parsed


def choose(bank, logits, alpha, embeddings):
    pooled, parsed = pool(bank, logits, alpha)
    if not bank:
        return [], pooled, []
    embeddings = np.asarray(embeddings)
    if embeddings.shape[:2] != (len(bank), 4) or not np.isfinite(embeddings).all():
        raise ValueError('Invalid candidate representations')
    if alpha == 0:
        selected = np.asarray(logits).argmax(axis=1).astype(int).tolist()
    else:
        selected = []
        for qi, row in enumerate(bank):
            ties = np.flatnonzero(pooled[qi] == pooled[qi].max()).tolist()
            def content_key(j):
                value = parsed[qi][j]
                return (value is None, value or '', row['candidates'][j]['text'],
                        np.asarray(embeddings[qi, j], dtype=np.float64).tobytes())
            selected.append(int(min(ties, key=content_key)))
    tie_rows = [i for i, row in enumerate(pooled) if int((row == row.max()).sum()) > 1]
    return selected, pooled, tie_rows
