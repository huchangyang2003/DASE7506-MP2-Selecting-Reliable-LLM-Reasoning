"""Meaningful numerical and boundary checks, without additional model fits."""
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="4"
import numpy as np
import reference_model as rm
from study_core import fit_pca
from experiment_io import ROOT,save

def main():
    rng=np.random.default_rng(20261030);x=rng.normal(size=(3,4,7));y=np.array([[1,0,1,0],[0,1,0,0],[1,1,1,1]],dtype=float)
    w=rng.normal(size=7);reg=np.full(7,.03);t=y/y.sum(1,keepdims=True)
    _,g=rm._loss_grad(w,x,t,reg)
    numeric=np.zeros_like(w);eps=1e-6
    for j in range(7):
        step=np.zeros(7);step[j]=eps
        numeric[j]=(rm._loss_grad(w+step,x,t,reg)[0]-rm._loss_grad(w-step,x,t,reg)[0])/(2*eps)
    assert np.allclose(g,numeric,atol=1e-8,rtol=1e-6)
    emb=rng.normal(size=(8,4,40));rows=np.arange(6);held=np.arange(6,8);p=fit_pca(emb,rows)
    emb[held]=1e6;other=fit_pca(emb,rows)
    assert np.array_equal(p["pca_mean"],other["pca_mean"]) and np.array_equal(p["pca_axes"],other["pca_axes"])
    save(ROOT/"verification/numerical_checks.json",{"reference_gradient_matches_finite_difference":True,"pca_heldout_invariance":True,
        "synthetic_checks_are_not_model_fits":True})
    print("Numerical and heldout-statistics checks passed")
if __name__=="__main__":main()
