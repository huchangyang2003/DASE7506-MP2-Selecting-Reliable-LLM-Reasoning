"""Recompute statistics against actual fit rows; run before validation."""
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="4"
import numpy as np
from experiment_io import ROOT,PROJECT,read,save,sha,utc,guard,load_split,disagreement,snapshot_original,source_snapshot
from provenance import snapshot
from study_core import reference_inputs,hand_inputs,tree_inputs,reference_scores,tree_scores
from submission import candidate_scores
import reference_model as rm
import lightgbm as lgb

def check_pca(emb,rows,path):
    with np.load(path,allow_pickle=False) as z:m={k:z[k].copy() for k in z.files}
    flat=emb[rows].reshape(-1,1024).astype(np.float64);mean=flat.mean(0)
    assert np.array_equal(mean,m["pca_mean"]);flat-=mean
    axes=m["pca_axes"];cov=flat.T@flat/len(flat)
    assert np.allclose(axes.T@axes,np.eye(32),atol=1e-10)
    residue=float(np.max(np.abs(cov@axes-axes*m["eigenvalues"])))
    assert residue<1e-8
    assert (np.diff(m["eigenvalues"])<=1e-10).all()
    return residue,m

def main():
    opened,denied=guard("train");out=ROOT/"runs/experiment";t=read(out/"training_complete.json");budget=read(out/"budget.json")
    assert t["no_validation_or_test_opened"] and sha(ROOT/"EXPERIMENT_PLAN.md")==budget["plan_sha256"]
    for name,digest in budget["sources"].items():assert sha(ROOT/"code"/name)==digest,name
    assert snapshot_original()==budget["original"] and snapshot()==budget["previous"]
    bank,arrays,y=load_split("train");hard=disagreement(bank);x=reference_inputs(bank,arrays);hand,names=hand_inputs(bank,arrays);emb=arrays["embeddings"]
    rows=read(out/"row_audit.json");assert len(rows)==10;opts=[];pca_errors=[];tree_checks=0
    for a in rows:
        pool=np.asarray(a["train"]);held=np.asarray(a["held"]);h=pool[hard[pool]]
        assert set(pool).isdisjoint(held) and set(pool)|set(held)==set(range(len(bank)))
        assert a["tree_and_pca_fit"]==pool.tolist() and a["reference_fit_and_stats"]==h.tolist()
        folder=out/f"seed{a['seed']}/fold{a['fold']}"
        scaler=rm.Scaler();scaler.fit(x[h])
        with np.load(folder/"reference_full/model.npz",allow_pickle=False) as z:
            assert np.array_equal(scaler.mean,z["mean"]) and np.array_equal(scaler.std,z["scale"])
            assert z["weights"].shape==(3,2129) and np.isfinite(z["weights"]).all()
        opts.extend(read(folder/"reference_full/optimizer.json"))
        error,_=check_pca(emb,pool,folder/"pca.npz");pca_errors.append(error)
        for path in folder.glob("tree_*/tree_training.json"):
            q=read(path);assert sha(path.parent/"tree.txt")==q["model_sha256"] and q["training_groups"]==len(pool) and q["completed_rounds"]==200
            tree_checks+=1
    assert tree_checks==60
    opts.extend(read(out/"reference_full/full_optimizer.json"))
    attempts=[q for o in opts for q in o["attempts"]];chosen=[o["attempts"][o["selected_attempt"]] for o in opts]
    assert len(attempts)==66 and len(chosen)==33
    full_rows=np.arange(len(bank));pca_error,pca=check_pca(emb,full_rows,out/"full_pca.npz");pca_errors.append(pca_error)
    for m in t["models"]:
        cp=ROOT/m["checkpoint"]
        for name,digest in m["sha256"].items():assert sha(cp/name)==digest
        if m["mode"]=="reference_full":
            scaler=rm.Scaler();scaler.fit(x[hard])
            with np.load(cp/"model.npz",allow_pickle=False) as z:
                mm={k:z[k].copy() for k in z.files}
                assert np.array_equal(scaler.mean,z["mean"]) and np.array_equal(scaler.std,z["scale"])
            expected=reference_scores(x[:9],mm)
        else:
            cfg=read(cp/"config.json");feat=hand
            if cfg["pca32"]:
                with np.load(cp/"transform.npz",allow_pickle=False) as z:
                    assert np.array_equal(z["pca_mean"],pca["pca_mean"]) and np.array_equal(z["pca_axes"],pca["pca_axes"])
                feat=tree_inputs(hand,emb,pca)
            expected=tree_scores(feat[:9],lgb.Booster(model_file=str(cp/"tree.txt")))
        actual=candidate_scores(bank[:9],{k:v[:9] for k,v in arrays.items()},cp)
        assert np.allclose(expected,actual,atol=1e-9),m["mode"]
    save(ROOT/"verification/pre_validation_audit.json",{"finished_utc":utc(),"before_validation":not(out/"validation").exists(),
        "reference_scalers_recomputed_exact":11,"pca_statistics_verified":11,"max_pca_eigen_residual":max(pca_errors),
        "fold_train_held_separation_checks":10,"tree_round_and_hash_checks":60,"deployment_matches_training_scores":True,
        "reference_optimizer":{"recorded_starts":66,"successful_starts":sum(a["success"] for a in attempts),
           "selected_members":33,"selected_successful":sum(a["success"] for a in chosen),
           "all_selected_successful":all(a["success"] for a in chosen),
           "all_parameters_finite":all(a["parameters_finite"] for a in attempts),
           "max_selected_gradient_inf":max(a["gradient_inf"] for a in chosen)},
        "old_artifacts_unchanged":True,"opened":sorted(opened),"denied":denied,"sources":source_snapshot()})
    print("Pre-validation audit passed; selected successful reference members:",sum(a["success"] for a in chosen),"/33")
if __name__=="__main__":main()
