import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="4"
import itertools
import numpy as np
from common import parse_answer,read_jsonl
from experiment_io import ROOT,PROJECT,DATA,read,save,sha,utc,guard,load_split,disagreement,metrics,snapshot_original,source_snapshot
from provenance import snapshot

def main():
    opened,denied=guard("train");out=ROOT/"diagnosis"
    if out.exists():raise FileExistsError("Never overwrite a diagnostic run")
    out.mkdir();orig=snapshot_original();prior=snapshot()
    save(out/"started.json",{"utc":utc(),"plan_sha256":sha(ROOT/"DIAGNOSTIC_PLAN.md"),"original":orig,"previous":prior,"sources":source_snapshot()})
    bank,arrays,y=load_split("train");hard=disagreement(bank);n=len(y)
    answers=np.array([[parse_answer(c["text"]) for c in b["candidates"]] for b in bank],dtype=object)
    result={};saved={}
    for seed in (20261022,20261029):
        paths={
            "original":(PROJECT/f"V3_OOF_Ensemble/runs/step4/complete_branch_oof_{seed}.npz","original"),
            "ce":(PROJECT/f"V4_Reference_Alignment/runs/scope/baseline/oof_{seed}.npz","scores"),
            "rank":(PROJECT/f"V3_Relative_Features/runs/step2/rank/oof_seed{seed}.npz","scores"),
            "zscore":(PROJECT/f"V3_Relative_Features/runs/step2/zscore/oof_seed{seed}.npz","scores"),
            "strict_arithmetic":(PROJECT/f"V3_Arithmetic_Reliability/runs/step3/strict_replace/oof_seed{seed}.npz","scores"),
            "hard_training":(PROJECT/f"V4_Reference_Alignment/runs/scope/hard_all_scaler/oof_{seed}.npz","scores")}
        scores={};fold=None;sources={}
        for mode,(path,key) in paths.items():
            with np.load(path,allow_pickle=False) as z:
                scores[mode]=z[key].copy()
                assert np.array_equal(z["labels"],y) and np.array_equal(z["hard"],hard)
                f=z["fold"]
                if fold is None:fold=f.copy()
                else:assert np.array_equal(fold,f),mode
            sources[mode]={"path":str(path.relative_to(PROJECT)),"sha256":sha(path)}
        selected={m:s.argmax(1) for m,s in scores.items()}
        correctness={m:y[np.arange(n),c].astype(bool) for m,c in selected.items()}
        picked={m:answers[np.arange(n),c] for m,c in selected.items()}
        a,b=correctness["original"],correctness["ce"]
        pair_any=a|b;family_any=np.any(np.stack(list(correctness.values())),axis=0)
        changed=selected["original"]!=selected["ce"];equivalent=(picked["original"]==picked["ce"]) & (picked["original"]!=None)
        pairs={}
        for m,k in itertools.combinations(scores,2):
            cm=selected[m]!=selected[k];eq=(picked[m]==picked[k])&(picked[m]!=None)
            pairs[m+"__"+k]={"different_index_hard":int((cm&hard).sum()),"same_answer_different_index_hard":int((cm&eq&hard).sum()),
                "different_answer_hard":int((cm&~eq&hard).sum()),"left_only_correct_hard":int((correctness[m]&~correctness[k]&hard).sum()),
                "right_only_correct_hard":int((correctness[k]&~correctness[m]&hard).sum())}
        distinct=np.array([len({a for a in row if a is not None}) for row in answers])
        support=np.array([max([list(row).count(a) for a in row if a is not None],default=0) for row in answers])
        strata={}
        for field,values in (("distinct_answers",distinct),("positive_candidates",y.sum(1).astype(int)),("largest_answer_support",support)):
            strata[field]={str(v):{"n":int(((values==v)&hard).sum()),"ce_correct":int((correctness["ce"]&(values==v)&hard).sum()),
                         "primary_both_wrong":int((~pair_any&(values==v)&hard).sum())} for v in sorted(set(values[hard]))}
        result[str(seed)]={"methods":{m:metrics(c,hard) for m,c in correctness.items()},"pair_oracle":metrics(pair_any,hard),
            "family_oracle":metrics(family_any,hard),"primary_both_wrong_hard":int((~pair_any&hard).sum()),
            "family_all_wrong_hard":int((~family_any&hard).sum()),"pairs":pairs,"strata":strata,"sources":sources}
        saved[seed]={"selected":selected,"correct":correctness,"picked":picked,"family_any":family_any,"fold":fold}
    stable=hard.copy()
    for s in saved:stable&=~saved[s]["correct"]["original"]&~saved[s]["correct"]["ce"]
    seed=20261022;rng=np.random.default_rng(20261030);used=set();cases=[]
    categories=(("stable_common_error",stable,16),
                ("complementary",hard&(saved[seed]["correct"]["original"]!=saved[seed]["correct"]["ce"]),8),
                ("same_answer_different_index",hard&(saved[seed]["selected"]["original"]!=saved[seed]["selected"]["ce"])&
                  (saved[seed]["picked"]["original"]==saved[seed]["picked"]["ce"])&(saved[seed]["picked"]["ce"]!=None),4))
    refs=read_jsonl(DATA/"gsm8k_train.jsonl")
    for name,mask,count in categories:
        available=[int(i) for i in np.flatnonzero(mask) if int(i) not in used]
        for i in sorted(rng.choice(available,min(count,len(available)),replace=False).tolist()):
            used.add(i);cases.append({"case":len(cases)+1,"category":name,"row":i,"id":bank[i]["id"],"question":bank[i]["question"],
                "reference":refs[i]["answer"],"labels":y[i].astype(int).tolist(),"candidates":bank[i]["candidates"],
                "selected":{str(s):{m:int(a[i]) for m,a in z["selected"].items()} for s,z in saved.items()},
                "correct":{str(s):{m:bool(a[i]) for m,a in z["correct"].items()} for s,z in saved.items()}})
    np.savez(out/"case_sets.npz",stable_common_error=stable,hard=hard,
             **{f"{s}_{m}":a for s,z in saved.items() for m,a in z["correct"].items()})
    save(out/"cases.json",cases)
    save(out/"summary.json",{"finished_utc":utc(),"questions":n,"hard_n":int(hard.sum()),"results":result,
        "stable_primary_both_wrong":int(stable.sum()),"sample_n":len(cases),"oracle_is_label_dependent_not_deployable":True,
        "old_artifacts_unchanged":snapshot_original()==orig and snapshot()==prior,"opened":sorted(opened),"denied":denied})
    print("Stable common errors:",int(stable.sum()),"; sampled cases:",len(cases))
    for s,z in result.items():print(s,"CE",z["methods"]["ce"]["disagreement"],"pair oracle",z["pair_oracle"]["disagreement"],
        "family oracle",z["family_oracle"]["disagreement"])
if __name__=="__main__":main()
