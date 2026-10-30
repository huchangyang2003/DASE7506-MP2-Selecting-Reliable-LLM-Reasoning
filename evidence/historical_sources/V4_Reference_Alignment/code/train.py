import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="4"
import argparse,time,zipfile
from pathlib import Path
import numpy as np
from common import PROTOCOL
from linear_core import build_inputs,fit_scaler,fit
from fusion_core import grouped_folds
from experiment_io import ROOT,DATA,read,save,sha,utc,guard,load_split,disagreement,metrics,snapshot_original,source_snapshot
from provenance import PREVIOUS,snapshot
from scope import MODES,fit_and_scaler_rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,default=ROOT/"runs/scope")
    out=ap.parse_args().output.resolve()
    if out.exists():raise FileExistsError("Use a new directory")
    out.mkdir(parents=True);opened,denied=guard("train")
    orig=snapshot_original();prior=snapshot();start=time.perf_counter()
    save(out/"budget.json",{"started_at_utc":utc(),"plan_sha256":sha(ROOT/"PLAN.md"),"sources":source_snapshot(),
         "original":orig,"previous":prior,"fit_budget":33,"modes":MODES,"seeds":[20261022,20261029]})
    bank,arrays,y=load_split("train");hard=disagreement(bank);x,names=build_inputs(bank,arrays)
    save(out/"inputs.json",{"questions":len(bank),"disagreement":int(hard.sum()),"feature_names":names,
         "data_sha256":{r:sha(DATA/r) for r in ("gsm8k_train.jsonl","candidates/gsm8k_train.jsonl","candidates/gsm8k_train.npz")}})
    folds={s:grouped_folds(bank,seed=s) for s in (20261022,20261029)}
    for seed,f in folds.items():save(out/f"folds_{seed}.json",[{"id":b["id"],"fold":int(z)} for b,z in zip(bank,f)])
    records=[];cv=[];models=[];audits=[]
    for mode in MODES:
        for seed,f in folds.items():
            scores=np.full(y.shape,np.nan)
            for fold in range(5):
                pool=np.flatnonzero(f!=fold);held=np.flatnonzero(f==fold)
                rows,stats=fit_and_scaler_rows(pool,hard,mode)
                assert not set(rows)&set(held) and not set(stats)&set(held)
                assert not {" ".join(bank[i]["question"].split()) for i in pool}&{" ".join(bank[i]["question"].split()) for i in held}
                mean,scale=fit_scaler(x[stats])
                tick=time.perf_counter()
                theta,opt=fit(np.ascontiguousarray((x[rows]-mean)/scale),y[rows],"uniform_ce",.01,.01)
                seconds=time.perf_counter()-tick
                folder=out/mode/f"seed{seed}"
                save(folder/f"fold{fold}_optimization.json",opt)
                if not opt["converged"]:raise RuntimeError("Unconverged fit")
                records.append({"mode":mode,"seed":seed,"fold":fold,"seconds":seconds,"fit_n":len(rows),
                    "statistics_n":len(stats),"converged":True,"gradient_inf":opt["gradient_inf"]})
                audits.append({"mode":mode,"seed":seed,"fold":fold,"pool":pool.tolist(),"fit":rows.tolist(),
                               "statistics":stats.tolist(),"held":held.tolist()})
                save(out/"fit_progress.json",records)
                scores[held]=((x[held]-mean)/scale)@theta[:-1]+theta[-1]
                print(f"{mode} seed={seed} fold={fold} fit={len(rows)} stats={len(stats)}",flush=True)
            np.savez(out/mode/f"oof_{seed}.npz",scores=scores,labels=y,hard=hard,fold=f)
            correct=y[np.arange(len(y)),scores.argmax(axis=1)].astype(bool)
            cv.append({"mode":mode,"seed":seed,**metrics(correct,hard)})
            if mode=="baseline":
                with np.load(PREVIOUS/"runs/step1"/("primary" if seed==20261022 else "confirmation")/"oof_scores.npz",allow_pickle=False) as z:
                    assert np.array_equal(scores,z["uniform_ce_e0.01_h0.01"]) and np.array_equal(f,z["fold"])
        rows,stats=fit_and_scaler_rows(np.arange(len(bank)),hard,mode)
        mean,scale=fit_scaler(x[stats]);tick=time.perf_counter()
        theta,opt=fit(np.ascontiguousarray((x[rows]-mean)/scale),y[rows],"uniform_ce",.01,.01)
        seconds=time.perf_counter()-tick
        save(out/mode/"full_optimization.json",opt)
        if not opt["converged"]:raise RuntimeError("Unconverged full fit")
        records.append({"mode":mode,"fold":"full","seconds":seconds,"fit_n":len(rows),"statistics_n":len(stats),
                        "converged":True,"gradient_inf":opt["gradient_inf"]})
        cp=out/mode/"checkpoint";cp.mkdir(parents=True)
        np.savez(cp/"model.npz",mean=mean,scale=scale,weights=theta[:-1],bias=np.asarray(theta[-1]))
        save(cp/"config.json",{"protocol":PROTOCOL,"feature_names":names,"method":"ce_training_scope",
             "mode":mode,"l2":.01,"objective":"uniform_ce","fit_split":"train",
             "fit_questions":len(rows),"statistics_questions":len(stats),"clipping":False})
        if mode=="baseline":
            with np.load(PREVIOUS/"runs/step1/uniform_ce_e0.01_h0.01/checkpoint/model.npz",allow_pickle=False) as z:
                for n,a in (("mean",mean),("scale",scale),("weights",theta[:-1]),("bias",np.asarray(theta[-1]))):
                    assert np.array_equal(a,z[n])
        with zipfile.ZipFile(cp/"model.npz") as z:size=sum(i.file_size for i in z.infolist())+(cp/"config.json").stat().st_size
        assert size<16*2**20
        models.append({"mode":mode,"checkpoint":str(cp.relative_to(ROOT)),"uncompressed_bytes":size,
                       "sha256":{n:sha(cp/n) for n in ("model.npz","config.json")}})
    assert len(records)==33 and snapshot_original()==orig and snapshot()==prior
    save(out/"row_audit.json",audits)
    save(out/"training_complete.json",{"finished_at_utc":utc(),"fit_count":33,"fits":records,"models":models,"cv":cv,
         "optimization_seconds":sum(r["seconds"] for r in records),"wall_seconds":time.perf_counter()-start,
         "opened":sorted(opened),"denied":denied,"no_validation_or_test_opened":True,"sources":source_snapshot()})
    print("TRAIN COMPLETE 33 fits",flush=True)
if __name__=="__main__":main()
