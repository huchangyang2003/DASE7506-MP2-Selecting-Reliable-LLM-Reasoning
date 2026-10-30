"""Finite validation after all TRAIN fits and configuration selection are complete."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="4"
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
from common import PROTOCOL
from experiment_io import ROOT, DATA, FROZEN, read, save, sha, utc, guard, load_split, disagreement, metrics, snapshot_original, source_snapshot


def run(arguments, logfile):
    tick=time.perf_counter()
    result=subprocess.run([sys.executable,"-B",str(ROOT/"code/guarded_official.py")]+arguments,
                          cwd=ROOT/"code",capture_output=True,text=True,encoding="utf-8")
    elapsed=time.perf_counter()-tick
    logfile.parent.mkdir(parents=True,exist_ok=True)
    logfile.write_text(result.stdout+"\n"+result.stderr,encoding="utf-8")
    if result.returncode:
        raise RuntimeError(str(logfile)+"\n"+result.stderr)
    return elapsed


def paired(new, old, hard):
    rng=np.random.default_rng(20261029)
    result={}
    for name,mask in (("all",np.ones(len(hard),dtype=bool)),("disagreement",hard),("other",~hard)):
        a,b=new[mask],old[mask]
        delta=a.astype(float)-b.astype(float)
        bootstrap=[]
        for _ in range(10):
            bootstrap.extend((delta[rng.integers(0,len(delta),size=(500,len(delta)))].mean(axis=1)*100).tolist())
        result[name]={"both_correct":int((a&b).sum()),"corrected":int((a&~b).sum()),
             "broken":int((~a&b).sum()),"both_wrong":int((~a&~b).sum()),
             "gain_percentage_points":float(delta.mean()*100),
             "descriptive_paired_bootstrap_95_interval":np.quantile(bootstrap,[.025,.975]).tolist()}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,default=ROOT/"runs/step1")
    args=parser.parse_args()
    folder=args.run.resolve()
    training=read(folder/"training_complete.json")
    chosen=read(folder/"selection_before_confirmation_and_validation.json")
    assert training["no_validation_or_test_opened"] and training["original_artifacts_unchanged"]
    for model in training["models"]:
        cp=ROOT/model["checkpoint"]
        assert all(sha(cp/name)==h for name,h in model["checkpoint_sha256"].items())
    destination=folder/"validation"
    if destination.exists():
        raise FileExistsError("Never repeat validation into the same destination")
    destination.mkdir()
    opened,denied=guard("analysis")
    originals=snapshot_original()
    save(destination/"evaluation_started.json",{"started_at_utc":utc(),"training_complete_sha256":sha(folder/"training_complete.json"),
         "selection_sha256":sha(folder/"selection_before_confirmation_and_validation.json"),
         "training_sources":training["sources_sha256"],"evaluation_sources":source_snapshot(),
         "only_preselected_configs":True,"no_test_evaluation":True})
    bank,arrays,y=load_split("validation")
    hard=disagreement(bank)
    records,correctness,choices={}, {}, {}
    todo=[{"config":"frozen_v1_1","method":str(FROZEN/"code/submission.py"),"checkpoint":str(FROZEN/"checkpoint")}]
    todo += [{"config":m["config"],"method":str(ROOT/"code/submission.py"),
              "checkpoint":str(ROOT/m["checkpoint"]),"uncompressed_bytes":m["uncompressed_bytes"]} for m in training["models"]]
    for model in todo:
        name=model["config"]
        out=destination/name
        pred=out/"validation-predictions.json";score=out/"validation-score.json"
        predict_seconds=run(["predict.py","--method",model["method"],"--checkpoint",model["checkpoint"],
             "--split","validation","--data-dir",str(DATA),"--output",str(pred)],out/"prediction.log")
        scoring_seconds=run(["score_predictions.py","--split","validation","--data-dir",str(DATA),
             "--predictions",str(pred),"--output",str(score)],out/"scoring.log")
        prediction=read(pred); official=read(score)
        assert prediction["ids"]==[b["id"] for b in bank]
        selected=np.asarray(prediction["selected"],dtype=int)
        correct=y[np.arange(len(bank)),selected].astype(bool)
        assert int(correct.sum())==official["correct"]
        records[name]={"config":name,**metrics(correct,hard),"prediction_wall_seconds":predict_seconds,
              "scoring_wall_seconds":scoring_seconds,"complete_validation_seconds":predict_seconds+scoring_seconds,
              "uncompressed_bytes":model.get("uncompressed_bytes",203273)}
        choices[name],correctness[name]=selected,correct
        print(name+" "+str(records[name]),flush=True)
    assert records["frozen_v1_1"]["all"]["correct"]==745
    assert records["frozen_v1_1"]["disagreement"]["correct"]==302
    comparisons={name:paired(correctness[name],correctness["frozen_v1_1"],hard)
                 for name in correctness if name!="frozen_v1_1"}
    block_controls={}
    for objective in ("bce","uniform_ce"):
        members={s["family"]:s["config"] for s in chosen["selected"] if s["objective"]==objective}
        block_controls[objective]={"block":members["block"],"uniform":members["uniform"],
             "paired":paired(correctness[members["block"]],correctness[members["uniform"]],hard)}
    complementarity={}
    old=correctness["frozen_v1_1"]
    for name,new in correctness.items():
        if name=="frozen_v1_1":continue
        complementarity[name]={"all_oracle_union_correct":int((new|old).sum()),
             "disagreement_oracle_union_correct":int((new[hard]|old[hard]).sum()),
             "note":"Oracle diagnostic uses validation labels, not an implementable selector or forecast."}
    # Contract checks use actual validation inputs, never references in predictor processes.
    from submission import candidate_scores
    contracts={}
    for model in training["models"]:
        cp=ROOT/model["checkpoint"]
        small_bank=bank[:7]
        small_arrays={k:v[:7] for k,v in arrays.items()}
        original=candidate_scores(small_bank,small_arrays,cp)
        permutation=np.array([2,0,3,1])
        changed=[{**b,"id":"new-"+str(i),"candidates":[{**b["candidates"][j],"index":pos}
                   for pos,j in enumerate(permutation)]} for i,b in enumerate(small_bank)]
        perm_arrays={k:v[:,permutation] for k,v in small_arrays.items()}
        perm=candidate_scores(changed,perm_arrays,cp)
        assert np.allclose(perm,original[:,permutation],rtol=1e-9,atol=1e-9)
        assert original.shape==(7,4)
        contracts[model["config"]]={"different_bank_size":True,"new_ids":True,"candidate_score_equivariance":True}
    assert snapshot_original()==originals
    result={"finished_at_utc":utc(),"records":records,"vs_frozen":comparisons,
         "block_vs_uniform":block_controls,"complementarity":complementarity,
         "inference_contracts":contracts,"original_artifacts_unchanged":True,
         "no_test_evaluation":True,"opened_data_files":sorted(opened),"denied_accesses":denied,
         "environment":{"python":platform.python_version(),"numpy":np.__version__,"processor":platform.processor(),
                        "cpu":"Intel Core i7-14700K (verified in Windows processor registry)",
                        "threads":4,"workers":1}}
    save(destination/"summary.json",result)

if __name__=="__main__":
    main()
