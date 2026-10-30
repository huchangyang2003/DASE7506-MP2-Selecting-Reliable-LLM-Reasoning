"""Meaningful numerical/contract checks for the new objectives and optimizer."""
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[k]="4"
import tempfile
from pathlib import Path
import numpy as np
from linear_core import loss_gradient, fit, fit_scaler, scores
from fusion_core import grouped_folds
from experiment_io import guard, save, ROOT


def main():
    rng=np.random.default_rng(20261029)
    # Cross the actual block boundary so both penalty gradients are checked.
    x=rng.normal(size=(5,4,1026))
    y=np.array([[1,0,0,0],[1,1,0,0],[1,1,1,1],[0,1,0,1],[0,0,1,0]],dtype=float)
    theta=rng.normal(scale=.01,size=1027)
    penalties=np.r_[np.full(1024,.1),np.full(2,.003)]
    checks={}
    for objective in ("bce","uniform_ce"):
        value,gradient=loss_gradient(theta,x,y,objective,penalties)
        errors=[]
        for i in (0,11,1023,1024,1025,1026):
            plus,minus=theta.copy(),theta.copy()
            plus[i]+=1e-5;minus[i]-=1e-5
            numerical=(loss_gradient(plus,x,y,objective,penalties)[0]-loss_gradient(minus,x,y,objective,penalties)[0])/2e-5
            errors.append(abs(numerical-gradient[i]))
        assert max(errors)<1e-7,(objective,errors)
        permutation=np.array([2,0,3,1])
        v,g=loss_gradient(theta,x[:,permutation],y[:,permutation],objective,penalties)
        assert np.allclose(v,value,atol=1e-12) and np.allclose(g,gradient,atol=1e-12)
        small=x[:,:,:6]
        fitted,record=fit(small,y,objective,.1,.1,max_iterations=2000)
        assert record["converged"],record
        assert record["final_loss"]<=loss_gradient(np.zeros(7),small,y,objective,np.full(6,.1))[0]+1e-10
        checks[objective]={"gradient_max_error":max(errors),"converged":True,"iterations":record["iterations"]}
    # Multiple positives use a uniform target, unlike positive-mass listwise.
    a=np.zeros((1,4,2));a[0,:,0]=np.log([.90,.09,.005,.005])
    target=np.array([[1,1,0,0]],dtype=float)
    loss,_=loss_gradient(np.array([1.,0.,0.]),a,target,"uniform_ce",np.zeros(2))
    assert abs(loss-(-.5*(np.log(.90)+np.log(.09))))<1e-12
    mean,scale=fit_scaler(np.array([[[1.,2.],[1.,3.],[1.,4.],[1.,5.]]]))
    assert scale[0]==1 and mean[0]==1
    bank=[{"question":" a  b ","candidates":[{"text":"#### 1"},{"text":"#### 2"},{"text":"#### 1"},{"text":"#### 2"}]},
          {"question":"a b","candidates":[{"text":"#### 1"}]*4}]
    for i in range(10):
        bank.append({"question":str(i),"candidates":[{"text":"#### 1"}]*4})
    folds=grouped_folds(bank,20261022)
    assert folds[0]==folds[1]
    opened,denied=guard("train")
    for name in ("gsm8k_test.jsonl","gsm8k_validation.npz"):
        try:
            Path(tempfile.gettempdir(),name).open("rb")
        except PermissionError:
            pass
        else:
            raise AssertionError("Guard did not deny "+name)
    result={"checks":checks,"multi_positive_ce_matches_definition":True,
            "candidate_permutation_invariance":True,"duplicate_question_grouping":True,
            "constant_scale_safe":True,"guard_denials_expected":denied,"passed":True}
    save(ROOT/"verification/numerical_checks.json",result)
    print(result)

if __name__=="__main__":
    main()
