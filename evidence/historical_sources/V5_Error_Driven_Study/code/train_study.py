import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="4"
import sys,time,zipfile
from pathlib import Path
import numpy as np
import scipy,lightgbm
from experiment_io import ROOT,PROJECT,DATA,read,save,sha,utc,guard,load_split,disagreement,metrics,snapshot_original,source_snapshot
from provenance import snapshot
from common import PROTOCOL,parse_answer
from fusion_core import grouped_folds
import reference_features as rf
from study_core import REFERENCE,TREE_MODES,SEEDS,fit_reference,reference_inputs,reference_scores,hand_inputs,fit_pca,tree_inputs,fit_tree,tree_scores,tree_order

def cp_record(mode,cp):
    size=0
    for p in cp.iterdir():
        if p.suffix==".npz":
            with zipfile.ZipFile(p) as z:size+=sum(i.file_size for i in z.infolist())
        else:size+=p.stat().st_size
    assert size<=16*2**20
    return {"mode":mode,"checkpoint":str(cp.relative_to(ROOT)),"uncompressed_bytes":size,
            "sha256":{p.name:sha(p) for p in sorted(cp.iterdir()) if p.is_file()}}

def main():
    opened,denied=guard("train");out=ROOT/"runs/experiment"
    if out.exists():raise FileExistsError("Never overwrite a training run")
    out.mkdir(parents=True);start=time.perf_counter();orig=snapshot_original();prior=snapshot()
    save(out/"budget.json",{"started_at_utc":utc(),"plan_sha256":sha(ROOT/"EXPERIMENT_PLAN.md"),
        "diagnostic_sha256":sha(ROOT/"diagnosis/summary.json"),"case_review_sha256":sha(ROOT/"diagnosis/CASE_REVIEW.json"),
        "original":orig,"previous":prior,"sources":source_snapshot(),"versions":{"python":sys.version,"numpy":np.__version__,
        "scipy":scipy.__version__,"lightgbm":lightgbm.__version__},"reference_member_fits":33,"scipy_starts":66,
        "tree_fits":61,"pca_fits":11})
    bank,arrays,y=load_split("train");hard=disagreement(bank);emb=arrays["embeddings"];n=len(y)
    assert all(rf.parse_answer(c["text"])==parse_answer(c["text"]) for b in bank for c in b["candidates"])
    x=reference_inputs(bank,arrays);hand,hand_names=hand_inputs(bank,arrays)
    assert np.all(x[:,:,24]==0) and np.all(x[:,:,28]==0)
    save(out/"inputs.json",{"questions":n,"disagreement":int(hard.sum()),"reference_shape":list(x.shape),"hand_shape":list(hand.shape),
        "official_parser_matches_reference_on_all_train":True,"faithful_grounding_zero_columns":[24,28],
        "data_sha256":{rel:sha(DATA/rel) for rel in ("gsm8k_train.jsonl","candidates/gsm8k_train.jsonl","candidates/gsm8k_train.npz")}})
    cv=[];records=[];audits=[];pca_records=[];baseline={}
    for seed in SEEDS:
        fold=grouped_folds(bank,seed=seed)
        with np.load(PROJECT/f"V4_Reference_Alignment/runs/scope/baseline/oof_{seed}.npz",allow_pickle=False) as z:
            assert np.array_equal(fold,z["fold"]) and np.array_equal(y,z["labels"])
            baseline[seed]=metrics(y[np.arange(n),z["scores"].argmax(1)].astype(bool),hard)
        allscores={m:np.full(y.shape,np.nan) for m in (REFERENCE,)+TREE_MODES}
        for f in range(5):
            pool=np.flatnonzero(fold!=f);held=np.flatnonzero(fold==f);hard_rows=pool[hard[pool]]
            assert not {" ".join(bank[i]["question"].split()) for i in pool}&{" ".join(bank[i]["question"].split()) for i in held}
            folder=out/f"seed{seed}/fold{f}";folder.mkdir(parents=True)
            audit={"seed":seed,"fold":f,"train":pool.tolist(),"held":held.tolist(),"reference_fit_and_stats":hard_rows.tolist(),"tree_and_pca_fit":pool.tolist()}
            audits.append(audit);save(out/"row_audit.json",audits)
            tick=time.perf_counter();model,opts=fit_reference(x,y,hard_rows,folder/REFERENCE)
            np.savez(folder/REFERENCE/"model.npz",**model)
            allscores[REFERENCE][held]=reference_scores(x[held],model)
            records.append({"mode":REFERENCE,"seed":seed,"fold":f,"seconds":time.perf_counter()-tick,"members":3,"starts":6})
            print("REFERENCE",seed,f,flush=True)
            tick=time.perf_counter();pca=fit_pca(emb,pool);np.savez(folder/"pca.npz",**pca)
            pca_records.append({"seed":seed,"fold":f,"seconds":time.perf_counter()-tick,"train_rows":len(pool),
                "sha256":sha(folder/"pca.npz")})
            xp=tree_inputs(hand,emb,pca)
            for mode in TREE_MODES:
                features=hand if "_hand_" in mode else xp
                tick=time.perf_counter();tree=fit_tree(features,y,pool,mode,folder/mode)
                allscores[mode][held]=tree_scores(features[held],tree)
                records.append({"mode":mode,"seed":seed,"fold":f,"seconds":time.perf_counter()-tick,"rounds":200})
            save(out/"fit_progress.json",records)
            print("TREE six configurations",seed,f,flush=True)
        for mode,scores in allscores.items():
            assert np.isfinite(scores).all()
            (out/mode).mkdir(exist_ok=True)
            np.savez(out/mode/f"oof_{seed}.npz",scores=scores,labels=y,hard=hard,fold=fold)
            correct=y[np.arange(n),scores.argmax(1)].astype(bool)
            cv.append({"mode":mode,"seed":seed,**metrics(correct,hard)})
        save(out/"cv_progress.json",cv)
    chosen=sorted(TREE_MODES,key=lambda m:tree_order(m,cv))[0]
    save(out/"tree_selection.json",{"selected":chosen,"selected_before_validation":True,"ranked_modes":sorted(TREE_MODES,key=lambda m:tree_order(m,cv)),
         "ordering":"mean hard correct, worst hard correct, mean overall correct, shallower, fewer features","cv":cv})
    models=[]
    rows=np.flatnonzero(hard);cp=out/REFERENCE/"checkpoint";cp.mkdir(parents=True)
    tick=time.perf_counter();model,opts=fit_reference(x,y,rows,cp)
    np.savez(cp/"model.npz",**model)
    save(cp/"config.json",{"protocol":PROTOCOL,"method":REFERENCE,"dims":[81,1024,1024],"members":3,"regularisation":[.001,.3,.3],
         "member_regularisation_scale":[1.,.5,2.],"fit_split":"train","fit_questions":len(rows),"clip":20.,
         "reference_commit":"7e3a2792b7b4442f5e53705a64b4560a2e365de9","known_grounding_type_issue_preserved":True})
    # optimizer diagnostics are evidence, not inference assets.
    (cp/"optimizer.json").replace(out/REFERENCE/"full_optimizer.json")
    records.append({"mode":REFERENCE,"fold":"full","seconds":time.perf_counter()-tick,"members":3,"starts":6})
    models.append(cp_record(REFERENCE,cp))
    tick=time.perf_counter();pca=fit_pca(emb,np.arange(n));np.savez(out/"full_pca.npz",**pca)
    pca_records.append({"fold":"full","seconds":time.perf_counter()-tick,"train_rows":n,"sha256":sha(out/"full_pca.npz")})
    cp=out/chosen/"checkpoint";cp.mkdir(parents=True)
    use_pca="_pca32_" in chosen;features=tree_inputs(hand,emb,pca) if use_pca else hand
    tick=time.perf_counter();tree=fit_tree(features,y,np.arange(n),chosen,cp)
    (cp/"tree_training.json").replace(out/chosen/"full_tree_training.json")
    np.savez(cp/"transform.npz",**({"pca_mean":pca["pca_mean"],"pca_axes":pca["pca_axes"]} if use_pca else {}))
    save(cp/"config.json",{"protocol":PROTOCOL,"method":"tree","mode":chosen,"pca32":use_pca,"hand_names":hand_names,
         "fit_split":"train","fit_questions":n,"feature_count":features.shape[-1],"rounds":200,"algorithm_seed":20261030})
    records.append({"mode":chosen,"fold":"full","seconds":time.perf_counter()-tick,"rounds":200})
    models.append(cp_record(chosen,cp))
    assert len(records)==72 and len(pca_records)==11
    assert snapshot_original()==orig and snapshot()==prior
    save(out/"training_complete.json",{"finished_at_utc":utc(),"models":models,"cv":cv,"baseline_cv":{str(k):v for k,v in baseline.items()},
        "tree_selected":chosen,"fits":records,"pca_fits":pca_records,"reference_member_fits":33,"scipy_starts":66,"tree_fits":61,
        "training_wall_seconds":time.perf_counter()-start,"optimizer_and_tree_seconds":sum(r["seconds"] for r in records),
        "pca_seconds":sum(r["seconds"] for r in pca_records),"opened":sorted(opened),"denied":denied,"no_validation_or_test_opened":True,
        "sources":source_snapshot()})
    print("TRAIN COMPLETE; selected tree",chosen,flush=True)
    for c in cv:print(c["mode"],c["seed"],c["disagreement"]["correct"],flush=True)
if __name__=="__main__":main()
