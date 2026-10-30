"""Post-hoc input diagnostics; no refit and no test access."""
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="4"
import numpy as np
from experiment_io import ROOT,DATA,read,save,guard,disagreement
from common import load_bank,parse_answer
def main():
    guard("analysis");bank=load_bank("validation",DATA);hard=disagreement(bank);result={};models={}
    for mode in ("baseline","hard_all_scaler","hard_hard_scaler"):
        p=read(ROOT/"runs/scope/validation"/mode/"validation-predictions.json")
        invalid=np.array([parse_answer(b["candidates"][j]["text"]) is None for b,j in zip(bank,p["selected"])])
        result[mode]={"invalid_chosen_all":int(invalid.sum()),"invalid_chosen_disagreement":int(invalid[hard].sum()),"invalid_chosen_other":int(invalid[~hard].sum())}
        with np.load(ROOT/"runs/scope"/mode/"checkpoint/model.npz",allow_pickle=False) as z:models[mode]={k:z[k].copy() for k in z.files}
    config=read(ROOT/"runs/scope/baseline/checkpoint/config.json");ratio=models["baseline"]["scale"]/models["hard_hard_scaler"]["scale"]
    details=[{"feature":config["feature_names"][i],"scale_ratio":float(ratio[i]),"all_scale":float(models["baseline"]["scale"][i]),"hard_scale":float(models["hard_hard_scaler"]["scale"][i])} for i in np.argsort(ratio)[-8:][::-1]]
    save(ROOT/"verification/posthoc_diagnostics.json",{"invalid_choices":result,"largest_scale_ratios":details,"no_refit_or_rule_changes":True,"not_a_causal_ablation":True})
if __name__=="__main__":main()
