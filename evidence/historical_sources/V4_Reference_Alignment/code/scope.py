import numpy as np
MODES=("baseline","hard_all_scaler","hard_hard_scaler")
def fit_and_scaler_rows(pool,hard,mode):
    if mode not in MODES:raise ValueError(mode)
    pool=np.asarray(pool,dtype=int)
    fit=pool if mode=="baseline" else pool[hard[pool]]
    stats=fit if mode=="hard_hard_scaler" else pool
    if not len(fit) or not len(stats):raise ValueError("Empty fit/statistics")
    assert set(fit)<=set(pool) and set(stats)<=set(pool)
    return fit,stats
