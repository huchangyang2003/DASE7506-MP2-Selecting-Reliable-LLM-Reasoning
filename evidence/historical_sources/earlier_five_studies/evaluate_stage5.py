"""Evaluate the complete predeclared budget, preserving all earlier stages."""
from datetime import datetime,timezone
import csv
import hashlib
import json
import os
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import numpy as np
from common import DATA,load_inputs,parse_answer,read_jsonl
from evaluate_stage4 import comparison
from methods import set_fusion
from no_test_access import install_guard
from run_stage1 import ROOT,SEEDS,run,save,summary
from train_stage5 import MODES
from verify_input_only import verify
from weighted_bce import disagreement_mask


def main():
    install_guard();destination=ROOT/'runs/stage5'
    if (destination/'summary.json').exists():raise FileExistsError('Stage 5 already evaluated')
    training=json.loads((destination/'training_complete.json').read_text())
    assert training['no_validation_or_test_opened']
    budget=json.loads((destination/'budget.json').read_text())
    for n,h in budget['source_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
    save(destination/'evaluation_started.json',{'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'training_finished_at_utc':training['finished_at_utc'],'evaluation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    bank,arrays=load_inputs('validation');refs=read_jsonl(DATA/'gsm8k_validation.jsonl')
    assert len(bank)==len(refs) and all(b['id']==r['id'] and b['question']==r['question'] for b,r in zip(bank,refs))
    labels=np.array([[int((a:=parse_answer(c['text'])) is not None and a==parse_answer(ref['answer']))
                      for c in b['candidates']] for b,ref in zip(bank,refs)],dtype=bool)
    hard=disagreement_mask(bank);records=[];choices={};correctness={}
    for model in training['models']:
        mode,seed=model['mode'],model['seed'];name=f'{mode}_seed{seed}';folder=destination/name
        prediction_seconds=run(['run_official.py','predict.py','--method','methods/set_fusion.py',
            '--checkpoint',str(ROOT/model['checkpoint']),'--split','validation','--output',str(folder/'validation-predictions.json')],
            folder/'prediction.log')
        scoring_seconds=run(['run_official.py','score_predictions.py','--split','validation','--predictions',
            str(folder/'validation-predictions.json'),'--output',str(folder/'validation-score.json')],folder/'scoring.log')
        output=json.loads((folder/'validation-predictions.json').read_text());score=json.loads((folder/'validation-score.json').read_text())
        chosen=np.asarray(output['selected']);correct=labels[np.arange(len(bank)),chosen]
        assert int(correct.sum())==score['correct']
        exact_match=None
        if mode=='control':
            old=json.loads((ROOT/f'runs/stage4_final/arithmetic_seed{seed}/validation-predictions.json').read_text())
            exact_match=output['selected']==old['selected'];assert exact_match
        choices[name],correctness[name]=chosen,correct
        record={'mode':mode,'seed':seed,**summary(correct,hard),'other_correct':int(correct[~hard].sum()),
            'other_n':int((~hard).sum()),'prediction_wall_seconds':prediction_seconds,'scoring_wall_seconds':scoring_seconds,
            'total_wall_seconds':prediction_seconds+scoring_seconds,'inference_uncompressed_bytes':model['inference_uncompressed_bytes'],
            'trainable_residual_parameters':model['trainable_residual_parameters'],'control_matches_stage4':exact_match}
        records.append(record);print(json.dumps(record),flush=True)
    aggregates=[{'mode':mode,'seeds':3,
        'accuracy_mean':float(np.mean([r['accuracy'] for r in records if r['mode']==mode])),
        'accuracy_sample_std':float(np.std([r['accuracy'] for r in records if r['mode']==mode],ddof=1)),
        'disagreement_mean':float(np.mean([r['disagreement_accuracy'] for r in records if r['mode']==mode])),
        'disagreement_sample_std':float(np.std([r['disagreement_accuracy'] for r in records if r['mode']==mode],ddof=1))} for mode in MODES]
    pairs=[(m,'control') for m in MODES if m!='control']+[('set16','unary16'),('set16','unary20')]
    comparisons={f'{m}_vs_{ref}':comparison(correctness,m,ref,hard) for m,ref in pairs}
    winner=max(aggregates,key=lambda r:(r['disagreement_mean'],r['accuracy_mean'],-MODES.index(r['mode'])))
    mode=winner['mode']
    practical=mode!='control' and comparisons[f'{mode}_vs_control']['disagreement']['mean_gain_percentage_points']>=1 and comparisons[f'{mode}_vs_control']['disagreement']['seed_wins']>=2
    selection={'best_average_mode':mode,'passes_minimum_practical_candidate_rule':practical,'primary_seed':20261022,
        'development_checkpoint':f'runs/stage5/{mode}_seed20261022/checkpoint','test_evaluated':False,
        'v1_or_stage4_replaced':False,'note':'Repeated validation development, not independent test. '
        'Neural seeds use corresponding historical base seed. Means across seeds are not inference ensembles.'}
    save(destination/'selection.json',selection);save(destination/'summary.json',records)
    save(destination/'aggregates.json',aggregates);save(destination/'paired_comparisons.json',comparisons)
    with (destination/'experiments.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    save(destination/'validation_diagnostics.json',[{'id':b['id'],'disagreement':bool(hard[i]),
        'correct':{n:bool(c[i]) for n,c in correctness.items()}} for i,b in enumerate(bank)])
    counts=np.asarray([len({a for c in b['candidates'] if (a:=parse_answer(c['text'])) is not None}) for b in bank])
    save(destination/'strata.json',[{'distinct_valid_answers':int(c),'n':int(np.sum(counts==c)),
        **{m:100*float(np.mean([correctness[f'{m}_seed{s}'][counts==c].mean() for s in SEEDS])) for m in MODES}}
        for c in sorted(set(counts))])
    old_diagnostics=json.loads((ROOT/'runs/stage4_final/validation_diagnostics.json').read_text())
    assert [r['id'] for r in old_diagnostics]==[r['id'] for r in bank]
    covered=np.asarray([r['checked_equations_by_candidate'] for r in old_diagnostics])>0
    bad=np.asarray([r['bad_equations_by_candidate'] for r in old_diagnostics])>0
    masks={'all_four_covered':covered.all(axis=1),'some_covered':covered.any(axis=1)&~covered.all(axis=1),
           'none_covered':~covered.any(axis=1),'some_candidate_has_bad':bad.any(axis=1)}
    save(destination/'coverage_strata.json',[{'subset':name,'n':int(mask.sum()),
        **{m:100*float(np.mean([correctness[f'{m}_seed{s}'][mask].mean() for s in SEEDS])) if mask.any() else None for m in MODES}}
        for name,mask in masks.items()])
    examples=[]
    for q,row in enumerate(bank):
        old,new=int(choices['control_seed20261022'][q]),int(choices['set16_seed20261022'][q])
        if old!=new:
            examples.append({'id':row['id'],'question':row['question'],'control_index':old,'set16_index':new,
                'control_correct':bool(labels[q,old]),'set16_correct':bool(labels[q,new]),
                'control_text':row['candidates'][old]['text'],'set16_text':row['candidates'][new]['text']})
    save(destination/'changed_choice_examples.json',examples)
    checks={}
    for m in dict.fromkeys(('set16',mode)):
        checks[m]=verify(set_fusion,bank,arrays,destination/f'{m}_seed20261022/checkpoint',choices[f'{m}_seed20261022'])
    save(destination/'inference_verification.json',checks)
    save(destination/'completion.json',{'finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'new_neural_fits':9,'official_validation_runs':12,'verification_passed':True,'test_evaluated':False})
    print(json.dumps({'aggregates':aggregates,'selection':selection},indent=2),flush=True)


if __name__=='__main__':main()
