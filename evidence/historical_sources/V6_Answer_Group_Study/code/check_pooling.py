"""Synthetic mathematical and interface checks; no course data or labels are loaded."""
from copy import deepcopy
import json
import numpy as np
from pooling_core import ALPHAS, choose, pool


def checks():
    def row(values):
        return {'id': 'synthetic', 'question': 'Synthetic only', 'candidates':
                [{'index': i, 'text': f'candidate-{i} #### {v}' if v is not None else f'unknown-{i}'}
                 for i, v in enumerate(values)]}
    bank = [row(['1/2', '0.5', '2', '3'])]
    logits = np.log(np.array([[0.95, 0.15, 0.60, 0.20]]))
    emb = np.arange(12, dtype=float).reshape(1, 4, 3)
    plain, p0, _ = choose(bank, logits, 0, emb)
    full, p1, _ = choose(bank, logits, 1, emb)
    assert plain == [0] and full == [2]
    assert p1[0, 0] == p1[0, 1]
    assert np.allclose(p1[0], [0.55/1.9, 0.55/1.9, 0.60/1.9, 0.20/1.9])
    assert abs(p1.sum() - 1) < 1e-15
    # Pooling can lower a high-scoring repeated answer rather than reward its votes.
    assert p1[0, 0] < p0[0, 0] and p1[0, 2] == p0[0, 2]
    invalid_bank = [row([None, None, '2', '2'])]
    invalid, _ = pool(invalid_bank, logits, 1)
    assert invalid[0, 0] == p0[0, 0] and invalid[0, 1] == p0[0, 1]
    assert choose(invalid_bank, logits, 1, emb)[0] == [0]
    all_invalid = [row([None]*4)]
    assert np.array_equal(pool(all_invalid, logits, 1)[0], p0)
    perm = np.array([2, 0, 3, 1])
    perm_bank = deepcopy(bank)
    perm_bank[0]['id'] = 'unseen'
    perm_bank[0]['candidates'] = [{**bank[0]['candidates'][j], 'index': k}
                                  for k, j in enumerate(perm)]
    for alpha in ALPHAS:
        selected, pooled, _ = choose(bank, logits, alpha, emb)
        assert np.allclose(pool(bank, logits + 500, alpha)[0], pooled, atol=2e-14, rtol=2e-14)
        changed, perm_pooled, _ = choose(perm_bank, logits[:, perm], alpha, emb[:, perm])
        assert np.array_equal(perm_pooled, pooled[:, perm])
        assert [pool(bank, logits, alpha)[1][0][selected[0]]] == \
               [pool(perm_bank, logits[:, perm], alpha)[1][0][changed[0]]]
    tie_bank = [row(['4', '3', '2', '1'])]
    tied = np.zeros((1, 4))
    assert choose(tie_bank, tied, 0, emb)[0] == [0]
    assert choose(tie_bank, tied, 0.5, emb)[0] == [3]
    duplicate = bank + deepcopy(bank)
    for alpha in ALPHAS:
        batch = choose(duplicate, np.repeat(logits, 2, axis=0), alpha, np.repeat(emb, 2, axis=0))[0]
        assert batch == choose(bank, logits, alpha, emb)[0]*2
        assert choose([], np.empty((0, 4)), alpha, np.empty((0, 4, 3)))[0] == []
    extreme = np.array([[1000., -1000., -500., 0.]])
    assert np.isfinite(pool(bank, extreme, 1)[0]).all()
    assert np.isclose(pool(bank, extreme, 1)[0].sum(), 1)
    return {'passed': True, 'checks': [
        'hand-computed mean pooling versus candidate maximum', 'fraction/decimal equivalence',
        'softmax shift invariance', 'candidate permutation', 'content tie-breaking',
        'a=0 legacy first-argmax ties', 'invalid candidates remain separate and selectable',
        'all-invalid scores unchanged', 'empty input', 'batch independence', 'extreme finite logits'],
        'scope': 'Synthetic only; no fitted model or course files read',
        'a0_different_answer_exact_ties': 'Preserves legacy position tie behavior by design.'}


if __name__ == '__main__':
    print(json.dumps(checks(), indent=2))
