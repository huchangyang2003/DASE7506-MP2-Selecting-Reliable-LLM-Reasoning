import numpy as np
from scope import fit_and_scaler_rows,MODES
from linear_core import fit_scaler
from experiment_io import ROOT,save
pool=np.array([0,1,2,3]);hard=np.array([True,False,True,False,True])
for mode in MODES:
    fit,stats=fit_and_scaler_rows(pool,hard,mode)
    assert 4 not in fit and 4 not in stats
    assert fit.tolist()==([0,1,2,3] if mode=="baseline" else [0,2])
    assert stats.tolist()==([0,2] if mode=="hard_hard_scaler" else [0,1,2,3])
    x=np.arange(5*4*2,dtype=float).reshape(5,4,2)
    before=fit_scaler(x[stats]);x[4]=1e12;after=fit_scaler(x[stats])
    assert all(np.array_equal(a,b) for a,b in zip(before,after))
save(ROOT/"verification/scope_checks.json",{"passed":True,"heldout_cannot_affect_statistics":True,"three_modes":True})
print("Scope checks passed")
