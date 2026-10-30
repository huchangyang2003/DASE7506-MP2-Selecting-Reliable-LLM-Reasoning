import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="4"
from pathlib import Path
import numpy as np
from experiment_io import ROOT,DATA,FROZEN,read,save,sha,utc,guard,load_split,disagreement,metrics,snapshot_original,source_snapshot
from provenance import PREVIOUS,snapshot
from scope import MODES
from validation_helpers import run,paired
def main():
    folder=ROOT/"runs/scope";t=read(folder/"training_complete.json")
    assert t["fit_count"]==33 and t["no_validation_or_test_opened"]
    for m in t["models"]:
        for n,d in m["sha256"].items():assert sha(ROOT/m["checkpoint"]/n)==d
    out=folder/"validation"
    if out.exists():raise FileExistsError("Do not overwrite")
    out.mkdir();opened,denied=guard("analysis");orig=snapshot_original();prior=snapshot(True)
    save(out/"started.json",{"started_at_utc":utc(),"training_complete_sha256":sha(folder/"training_complete.json"),"sources":source_snapshot()})
    bank,arrays,y=load_split("validation");hard=disagreement(bank)
    todo=[{"mode":"frozen_v1_1","method":str(FROZEN/"code/submission.py"),"checkpoint":str(FROZEN/"checkpoint"),"uncompressed_bytes":203273}]
    todo.extend({**m,"method":str(ROOT/"code/submission.py"),"checkpoint":str(ROOT/m["checkpoint"])} for m in t["models"])
    records={};correctness={};choices={}
    for m in todo:
        mode=m["mode"];dest=out/mode
        pred=dest/"validation-predictions.json";score=dest/"validation-score.json"
        ps=run(["predict.py","--method",m["method"],"--checkpoint",m["checkpoint"],"--split","validation","--data-dir",str(DATA),"--output",str(pred)],dest/"prediction.log")
        ss=run(["score_predictions.py","--split","validation","--data-dir",str(DATA),"--predictions",str(pred),"--output",str(score)],dest/"scoring.log")
        p=read(pred);s=read(score);choice=np.asarray(p["selected"])
        assert p["ids"]==[b["id"] for b in bank]
        c=y[np.arange(len(y)),choice].astype(bool);assert int(c.sum())==s["correct"]
        records[mode]={"mode":mode,**metrics(c,hard),"uncompressed_bytes":m["uncompressed_bytes"],"prediction_seconds":ps,"scoring_seconds":ss,"wall_seconds":ps+ss}
        correctness[mode]=c;choices[mode]=choice
        print(mode+" "+str(records[mode]),flush=True)
    assert choices["baseline"].tolist()==read(PREVIOUS/"runs/step1/validation/uniform_ce_e0.01_h0.01/validation-predictions.json")["selected"]
    assert records["frozen_v1_1"]["all"]["correct"]==745
    from submission import candidate_scores
    for m in t["models"]:
        small=bank[:7];a={k:v[:7] for k,v in arrays.items()};perm=np.array([2,0,3,1])
        p=candidate_scores(small,a,ROOT/m["checkpoint"])
        changed=[{**b,"id":"new-"+str(i),"candidates":[{**b["candidates"][j],"index":pos} for pos,j in enumerate(perm)]} for i,b in enumerate(small)]
        z=candidate_scores(changed,{k:v[:,perm] for k,v in a.items()},ROOT/m["checkpoint"])
        assert np.allclose(z,p[:,perm],atol=1e-9,rtol=1e-9)
    assert snapshot_original()==orig and snapshot(True)==prior
    save(out/"summary.json",{"finished_at_utc":utc(),"records":records,"paired":{m:paired(correctness[m],correctness["baseline"],hard) for m in MODES[1:]},
         "input_contracts_passed":True,"baseline_choices_exact":True,"prior_unchanged":True,"opened":sorted(opened),"denied":denied,
         "no_test_evaluation":True,"cpu":"Intel Core i7-14700K, 4 BLAS threads, 1 worker"})
if __name__=="__main__":main()
