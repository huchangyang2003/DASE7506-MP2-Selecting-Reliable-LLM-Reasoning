"""Frozen recipe definitions and grouped train-only estimators."""
import numpy as np
from scipy.optimize import minimize
import lightgbm as lgb
import reference_features as rf
import reference_model as rm
from common import parse_answer
from linear_core import build_inputs
from experiment_io import save,sha

TREE_MODES=tuple(f"tree_{kind}_d{depth}" for kind in ("hand","pca32") for depth in (2,3,4))
REFERENCE="reference_full"
SEEDS=(20261022,20261029)
REFERENCE_SEED=20251021
REFERENCE_SCALES=(1.,.5,2.)
PARAMS={"objective":"lambdarank","metric":"ndcg","ndcg_eval_at":[1],"label_gain":[0,1],
        "lambdarank_truncation_level":1,"learning_rate":.05,"min_data_in_leaf":40,"lambda_l2":1.,
        "max_bin":63,"feature_fraction":1.,"bagging_fraction":1.,"bagging_freq":0,"verbosity":-1,
        "deterministic":True,"force_col_wise":True,"num_threads":4,"seed":20261030,
        "feature_fraction_seed":20261030,"bagging_seed":20261030,"data_random_seed":20261030}
ROUNDS=200

def reference_inputs(bank,arrays):
    # Same answer contract; mirrored parser is checked separately, not used to score.
    rf.parse_answer=parse_answer
    parts=rf.build_features(bank,arrays)
    x=np.concatenate(parts,axis=-1)
    assert x.shape==(len(bank),4,2129) and np.isfinite(x).all()
    return x

def hand_inputs(bank,arrays):
    x,names=build_inputs(bank,arrays)
    return x[:,:,1024:].astype(np.float64),names[1024:]

def fit_pca(embedded,rows):
    flat=embedded[rows].reshape(-1,embedded.shape[-1]).astype(np.float64)
    mean=flat.mean(axis=0);flat-=mean
    cov=flat.T@flat/len(flat)
    values,vectors=np.linalg.eigh(cov)
    axes=vectors[:,-32:][:,::-1].copy()
    for j in range(axes.shape[1]):
        i=np.argmax(np.abs(axes[:,j]))
        if axes[i,j]<0:axes[:,j]*=-1
    assert np.allclose(axes.T@axes,np.eye(32),atol=1e-10)
    return {"pca_mean":mean,"pca_axes":axes,"eigenvalues":values[-32:][::-1]}

def tree_inputs(hand,embedded,pca=None):
    if pca is None:return hand
    projected=(embedded.astype(np.float64)-pca["pca_mean"])@pca["pca_axes"]
    return np.concatenate([hand,projected],axis=-1)

def fit_reference(x,y,rows,folder):
    scaler=rm.Scaler();scaler.fit(x[rows]);z=scaler.apply(x[rows]);target=y[rows]/y[rows].sum(1,keepdims=True)
    reg=np.concatenate([np.full(81,.001),np.full(1024,.3),np.full(1024,.3)])
    weights=[];opts=[]
    for member,mult in enumerate(REFERENCE_SCALES):
        rng=np.random.default_rng(REFERENCE_SEED+member);attempts=[];best=None
        for attempt in range(2):
            w0=rng.normal(0.,1e-3,x.shape[-1]) if attempt==0 else np.zeros(x.shape[-1])
            res=minimize(rm._loss_grad,w0,args=(z,target,reg*mult,None,None,1.),jac=True,method="L-BFGS-B",
                         options={"maxiter":600,"maxfun":1200})
            loss,grad=rm._loss_grad(res.x,z,target,reg*mult)
            rec={"member":member,"regularisation_multiplier":mult,"attempt":attempt,"success":bool(res.success),
                 "message":str(res.message),"iterations":int(res.nit),"evaluations":int(res.nfev),"loss":float(loss),
                 "gradient_inf":float(np.max(np.abs(grad))),"parameters_finite":bool(np.isfinite(res.x).all())}
            attempts.append(rec)
            if best is None or loss<best[0]:best=(loss,res.x.copy(),attempt)
        assert all(a["parameters_finite"] for a in attempts),attempts
        weights.append(best[1]);opts.append({"selected_attempt":best[2],"attempts":attempts})
        save(folder/"optimizer.json",opts)
        print("  reference member",member,"stops",[a["success"] for a in attempts],flush=True)
    save(folder/"optimizer.json",opts)
    return {"mean":scaler.mean,"scale":scaler.std,"weights":np.stack(weights)},opts

def reference_scores(x,model):
    z=np.clip((x.astype(np.float64)-model["mean"])/model["scale"],-20.,20.)
    return np.mean([z@w for w in model["weights"]],axis=0)

def fit_tree(x,y,rows,mode,folder):
    depth=int(mode.rsplit("d",1)[1]);params={**PARAMS,"max_depth":depth,"num_leaves":2**depth}
    d=lgb.Dataset(np.ascontiguousarray(x[rows].reshape(-1,x.shape[-1])),label=y[rows].reshape(-1).astype(np.int32),
                  group=np.full(len(rows),4,dtype=np.int32),free_raw_data=True)
    booster=lgb.train(params,d,num_boost_round=ROUNDS)
    folder.mkdir(parents=True,exist_ok=True);path=folder/"tree.txt";booster.save_model(str(path))
    assert booster.current_iteration()==ROUNDS
    assert np.isfinite(booster.predict(x[rows[:7]].reshape(-1,x.shape[-1]),num_threads=4)).all()
    save(folder/"tree_training.json",{"mode":mode,"params":params,"rounds":ROUNDS,"completed_rounds":booster.current_iteration(),
        "feature_count":x.shape[-1],"training_groups":len(rows),"model_sha256":sha(path)})
    return booster

def tree_scores(x,model):
    return model.predict(np.ascontiguousarray(x.reshape(-1,x.shape[-1])),num_threads=4).reshape(x.shape[:2])

def tree_order(mode,cv):
    a=[z for z in cv if z["mode"]==mode]
    return (-float(np.mean([r["disagreement"]["correct"] for r in a])),
            -min(r["disagreement"]["correct"] for r in a),
            -float(np.mean([r["all"]["correct"] for r in a])),
            int(mode.rsplit("d",1)[1]),mode!="tree_hand_d"+mode[-1])
