"""Predeclared TRAIN-only CV, four family finalists, confirmation and refit."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key] = "4"
import argparse
from pathlib import Path
import time
import zipfile
import numpy as np
from common import PROTOCOL
from fusion_core import grouped_folds
from linear_core import build_inputs, fit_scaler, fit, scores
from experiment_io import ROOT, DATA, read, save, sha, utc, guard, load_split, disagreement, metrics, snapshot_original, source_snapshot


def configuration(objective, emb, hand):
    return {"objective":objective,"embedding_l2":emb,"hand_l2":hand}


def key_for(config):
    return config["objective"]+"_e"+str(config["embedding_l2"])+"_h"+str(config["hand_l2"])


def grid():
    entries = {}
    for objective in ("bce","uniform_ce"):
        for value in (0.001,0.01,0.1):
            config=configuration(objective,value,value)
            entries.setdefault(key_for(config),{**config,"families":[]})["families"].append("uniform")
        for emb in (0.01,0.1,0.3):
            for hand in (0.001,0.01):
                config=configuration(objective,emb,hand)
                entries.setdefault(key_for(config),{**config,"families":[]})["families"].append("block")
    return entries


def cv(x, y, hard, bank, entries, fold_seed, destination, record_fits):
    folds = grouped_folds(bank, seed=fold_seed)
    save(destination/"fold_assignments.json",[{"id":b["id"],"fold":int(f)} for b,f in zip(bank,folds)])
    oof={name:np.full(y.shape,np.nan) for name in entries}
    fold_audit=[]
    for fold in range(5):
        train, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        train_questions={" ".join(bank[i]["question"].split()) for i in train}
        held_questions={" ".join(bank[i]["question"].split()) for i in held}
        assert not train_questions & held_questions
        mean, scale = fit_scaler(x[train])
        x_train=np.ascontiguousarray((x[train]-mean)/scale)
        x_held=np.ascontiguousarray((x[held]-mean)/scale)
        fold_audit.append({"fold":fold,"train_questions":len(train),"held_questions":len(held),
                           "train_disagreement":int(hard[train].sum()),"held_disagreement":int(hard[held].sum()),
                           "normalized_question_overlap":0,"train_rows":train.tolist(),"held_rows":held.tolist()})
        for name,config in entries.items():
            started=time.perf_counter()
            theta, record=fit(x_train,y[train],config["objective"],config["embedding_l2"],config["hand_l2"])
            elapsed=time.perf_counter()-started
            save(destination/name/("fold_"+str(fold)+"_optimization.json"),record)
            record_fits.append({"partition_seed":fold_seed,"config":name,"fold":fold,"seconds":elapsed,
                                "converged":record["converged"],"iterations":record["iterations"],
                                "gradient_inf":record["gradient_inf"]})
            save(destination/"fit_progress.json",record_fits)
            if not record["converged"]:
                raise RuntimeError("Non-converged fit: "+name+" fold "+str(fold))
            oof[name][held]=x_held@theta[:-1]+theta[-1]
            print(f"{fold_seed} fold={fold} {name} iterations={record['iterations']} seconds={elapsed:.3f}",flush=True)
    save(destination/"fold_audit.json",fold_audit)
    np.savez(destination/"oof_scores.npz",fold=folds,hard=hard,labels=y,**oof)
    summaries=[]
    for name,values in oof.items():
        assert np.isfinite(values).all()
        correct=y[np.arange(len(y)),values.argmax(axis=1)].astype(bool)
        summaries.append({"config":name,**entries[name],**metrics(correct,hard),
                          "by_fold":[{"fold":f,**metrics(correct[folds==f],hard[folds==f])} for f in range(5)]})
    save(destination/"summary.json",summaries)
    return summaries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"runs/step1")
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError("Never overwrite a completed or partial experiment")
    output.mkdir(parents=True)
    opened,denied=guard("train")
    started=time.perf_counter()
    originals=snapshot_original()
    entries=grid()
    save(output/"budget.json",{"started_at_utc":utc(),"plan_sha256":sha(ROOT/"PLAN.md"),
         "sources_sha256":source_snapshot(),"original_artifacts_before":originals,
         "grid":entries,"distinct_configurations":len(entries),"maximum_fits":104,
         "primary_fold_seed":20261022,"confirmation_fold_seed":20261029,
         "thread_limits":{k:os.environ[k] for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")}})
    bank,arrays,y=load_split("train")
    hard=disagreement(bank)
    feature_tick=time.perf_counter()
    x,names=build_inputs(bank,arrays)
    feature_seconds=time.perf_counter()-feature_tick
    save(output/"inputs.json",{"questions":len(bank),"disagreement":int(hard.sum()),"positive_labels":int(y.sum()),
         "dimension":x.shape[-1],"feature_names":names,"feature_seconds":feature_seconds,
         "train_data_sha256":{rel:sha(DATA/rel) for rel in ("gsm8k_train.jsonl","candidates/gsm8k_train.jsonl","candidates/gsm8k_train.npz")}})
    fits=[]
    primary=cv(x,y,hard,bank,entries,20261022,output/"primary",fits)
    selected=[]
    for objective in ("bce","uniform_ce"):
        for family in ("uniform","block"):
            eligible=[s for s in primary if s["objective"]==objective and family in s["families"]]
            best=max(eligible,key=lambda s:(s["disagreement"]["correct"],s["all"]["correct"],s["embedding_l2"],s["hand_l2"]))
            selected.append({"objective":objective,"family":family,"config":best["config"],
                             "primary_oof":{k:best[k] for k in ("all","disagreement","other")}})
    save(output/"selection_before_confirmation_and_validation.json",{"selected_at_utc":utc(),"selected":selected,
         "rule":"Primary TRAIN OOF disagreement correct, all correct, stronger embedding L2, stronger hand L2",
         "validation_not_opened":True,"test_not_opened":True,"sources_sha256":source_snapshot()})
    selected_names={s["config"] for s in selected}
    finalists={name:config for name,config in entries.items() if name in selected_names}
    confirmation=cv(x,y,hard,bank,finalists,20261029,output/"confirmation",fits)
    mean,scale=fit_scaler(x)
    standardized=np.ascontiguousarray((x-mean)/scale)
    full_records=[]
    for name,config in finalists.items():
        tick=time.perf_counter()
        theta,optimization=fit(standardized,y,config["objective"],config["embedding_l2"],config["hand_l2"])
        seconds=time.perf_counter()-tick
        save(output/name/"optimization.json",optimization)
        if not optimization["converged"]:
            raise RuntimeError("Full training failed")
        checkpoint=output/name/"checkpoint"
        checkpoint.mkdir(parents=True)
        np.savez(checkpoint/"model.npz",mean=mean,scale=scale,weights=theta[:-1],bias=np.asarray(theta[-1]))
        save(checkpoint/"config.json",{"protocol":PROTOCOL,"method":"v3_linear_step1","feature_names":names,
             **config,"fit_split":"train","uses_validation_for_fit":False,"arith_centered":True,
             "clipping":False,"standardization":"TRAIN candidate mean/population std, std<1e-8 -> 1",
             "intercept":"unpenalized" if config["objective"]=="bce" else "fixed_zero",
             "first_tie_argmax":True})
        with zipfile.ZipFile(checkpoint/"model.npz") as z:
            uncompressed=sum(i.file_size for i in z.infolist())+(checkpoint/"config.json").stat().st_size
        assert uncompressed <= 16*2**20
        full_records.append({"config":name,"seconds":seconds,"converged":True,
             "checkpoint":str(checkpoint.relative_to(ROOT)),"uncompressed_bytes":uncompressed,
             "checkpoint_sha256":{n:sha(checkpoint/n) for n in ("model.npz","config.json")}})
        fits.append({"config":name,"fold":"full_train","seconds":seconds,"converged":True,
                     "iterations":optimization["iterations"],"gradient_inf":optimization["gradient_inf"]})
        print("Full refit "+name,flush=True)
    assert snapshot_original()==originals
    save(output/"training_complete.json",{"finished_at_utc":utc(),"selected":selected,
         "confirmation":confirmation,"models":full_records,"fits":fits,
         "fit_count":len(fits),"optimization_seconds":sum(r["seconds"] for r in fits),
         "training_wall_seconds":time.perf_counter()-started,"opened_data_files":sorted(opened),
         "denied_accesses":denied,"no_validation_or_test_opened":True,"original_artifacts_unchanged":True,
         "sources_sha256":source_snapshot(),"feature_seconds":feature_seconds})
    print("TRAIN COMPLETE "+str(len(fits))+" fits",flush=True)

if __name__=="__main__":
    main()
