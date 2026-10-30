import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="4"
import numpy as np
from experiment_io import ROOT,PROJECT,DATA,FROZEN,read,save,sha,utc,guard,load_split,disagreement,metrics,snapshot_original,source_snapshot
from provenance import snapshot
from validation_helpers import run,paired
from submission import candidate_scores

def main():
    out=ROOT/"runs/experiment";t=read(out/"training_complete.json")
    assert t["no_validation_or_test_opened"] and t["tree_fits"]==61 and t["scipy_starts"]==66
    for model in t["models"]:
        for name,h in model["sha256"].items():assert sha(ROOT/model["checkpoint"]/name)==h
    val=out/"validation"
    if val.exists():raise FileExistsError("Never overwrite evaluation")
    val.mkdir();opened,denied=guard("analysis");orig=snapshot_original();prior=snapshot(True)
    save(val/"started.json",{"started_at_utc":utc(),"training_complete_sha256":sha(out/"training_complete.json"),"sources":source_snapshot(),
        "selected_tree":t["tree_selected"],"plan_sha256":sha(ROOT/"EXPERIMENT_PLAN.md")})
    bank,arrays,y=load_split("validation");hard=disagreement(bank);records={};correctness={}
    ce=PROJECT/"V4_Reference_Alignment"
    todo=[{"mode":"frozen_v1_1","method":str(FROZEN/"code/submission.py"),"checkpoint":str(FROZEN/"checkpoint"),"uncompressed_bytes":203273},
          {"mode":"ce_baseline","method":str(ce/"code/submission.py"),"checkpoint":str(ce/"runs/scope/baseline/checkpoint"),"uncompressed_bytes":50244}]
    todo.extend({**m,"method":str(ROOT/"code/submission.py"),"checkpoint":str(ROOT/m["checkpoint"])} for m in t["models"])
    contracts={}
    for m in todo:
        mode=m["mode"];folder=val/mode;pred=folder/"validation-predictions.json";score=folder/"validation-score.json"
        ps=run(["predict.py","--method",m["method"],"--checkpoint",m["checkpoint"],"--split","validation","--data-dir",str(DATA),"--output",str(pred)],folder/"prediction.log")
        ss=run(["score_predictions.py","--split","validation","--data-dir",str(DATA),"--predictions",str(pred),"--output",str(score)],folder/"scoring.log")
        p,s=read(pred),read(score);c=y[np.arange(len(y)),p["selected"]].astype(bool)
        assert p["ids"]==[b["id"] for b in bank] and int(c.sum())==s["correct"]
        correctness[mode]=c;records[mode]={"mode":mode,**metrics(c,hard),"uncompressed_bytes":m["uncompressed_bytes"],
             "prediction_seconds":ps,"scoring_seconds":ss,"wall_seconds":ps+ss}
        print(mode,records[mode]["all"],records[mode]["disagreement"],flush=True)
        if mode=="ce_baseline":assert p["selected"]==read(ce/"runs/scope/validation/baseline/validation-predictions.json")["selected"]
        if mode=="frozen_v1_1":assert int(c.sum())==745
        if mode in [z["mode"] for z in t["models"]]:
            small=bank[:7];a={k:v[:7] for k,v in arrays.items()};perm=np.array([2,0,3,1]);cp=m["checkpoint"]
            before=candidate_scores(small,a,cp)
            changed=[{**b,"id":"never-seen-"+str(i),"candidates":[{**b["candidates"][j],"index":pos} for pos,j in enumerate(perm)]} for i,b in enumerate(small)]
            after=candidate_scores(changed,{k:v[:,perm] for k,v in a.items()},cp)
            delta=float(np.max(np.abs(after-before[:,perm])))
            assert np.allclose(after,before[:,perm],atol=5e-5,rtol=1e-6),(mode,delta)
            assert candidate_scores([],{"embeddings":np.empty((0,4,1024))},cp).shape==(0,4)
            for count in (1,3):assert np.allclose(candidate_scores(small[:count],{k:v[:count] for k,v in a.items()},cp),before[:count],atol=1e-7)
            contracts[mode]={"new_ids_candidate_permutation_and_batches":[1,3,7],"empty_bank":True,"max_permutation_score_difference":delta}
    assert snapshot_original()==orig and snapshot(True)==prior
    save(val/"summary.json",{"finished_utc":utc(),"records":records,"paired":{m:paired(correctness[m],correctness["ce_baseline"],hard) for m in contracts},
         "input_contracts":contracts,"references_reproduced":True,"prior_unchanged":True,"opened":sorted(opened),"denied":denied,
         "no_test_evaluation":True,"cpu":"Intel Core i7-14700K; 4 total BLAS/OpenMP threads; one model worker"})
if __name__=="__main__":main()
