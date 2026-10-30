"""Four frozen validation evaluations and predeclared test-eligibility decision; no fit/test."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
PROJECT=ROOT.parent
OLD=PROJECT/'V3_Block_Regularization'
sys.path.insert(0,str(OLD/'code'))
from common import PROTOCOL,parse_answer
from experiment_io import DATA,guard,load_split,disagreement,metrics
OPENED,DENIED=guard('analysis')
PREDICT_CALLS=0
SCORE_CALLS=0


def validation_bank_only(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.name.lower().startswith('gsm8k_') and not path.name.lower().startswith('gsm8k_validation'):
            raise PermissionError('StageD only accesses validation bank files')


sys.addaudithook(validation_bank_only)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def snapshot():
    result={}
    for base in (ROOT,OLD):
        manifest=base/'ARTIFACT_MANIFEST.json'
        result[str(manifest)]=sha(manifest)
        for relative,entry in read(manifest)['files'].items():
            p=base/relative;digest=sha(p)
            assert digest==entry['sha256'],str(p)
            result[str(p)]=digest
    frozen=PROJECT/'Final_Optimized_Solution'
    for relative,entry in read(frozen/'FREEZE.json')['files'].items():
        p=frozen/relative;digest=sha(p);assert digest==entry['sha256'];result[str(p)]=digest
    p=frozen/'output/pdf/DASE7506_MP2_Final_Report.pdf';result[str(p)]=sha(p)
    publication=PROJECT/'First_Publication_20261022/DASE7506-MP2-Selecting-Reliable-LLM-Reasoning'
    pub=read(publication/'PUBLICATION_MANIFEST.json')
    p=publication/'PUBLICATION_MANIFEST.json';result[str(p)]=sha(p)
    for relative in ('checkpoint/model.npz','checkpoint/config.json','REPORT.pdf'):
        p=publication/relative;digest=sha(p)
        assert digest==pub['files'][relative]['sha256'],relative
        result[str(p)]=digest
    return result


def assets(cp):
    size=0
    for p in cp.rglob('*'):
        if p.is_file():
            if p.suffix=='.npz':
                with zipfile.ZipFile(p)as z:size+=sum(i.file_size for i in z.infolist())
            else:size+=p.stat().st_size
    return size


def candidates():
    status=read(ROOT/'STATUS.json')
    assert status['stage_c']=='completed' and status['stage_d']=='not_started'
    assert status['normal_fit_budget_remaining']==0 and status['normal_validation_budget_remaining']==4
    a=read(ROOT/'runs/pooling/stage_a/summary.json')
    b=read(ROOT/'runs/group_models/stage_b/summary.json')
    c=read(ROOT/'runs/subset_augmentation/stage_c/summary.json')
    assert a['selection']['selected_alpha']==1 and b['selection']['selected_group_mode']=='group_mean_context'
    assert read(ROOT/'runs/pooling/stage_a/deployment_checks_thread_capped.json')['passed']
    assert read(ROOT/'runs/group_models/stage_b/audit.json')['passed']
    assert read(ROOT/'runs/subset_augmentation/stage_c/audit.json')['passed']
    full={v['mode']:v for v in b['full_models']}
    cp=ROOT/a['selected_checkpoint']
    assert read(cp/'config.json')['alpha']==1 and read(cp/'config.json')['temperature']==1
    assert read(cp/'config.json')['selection_sha256']==sha(ROOT/'runs/pooling/stage_a/selection_before_confirmation.json')
    for name in ('model.npz','config.json'):
        assert sha(cp/'baseline'/name)==sha(OLD/'runs/step1/uniform_ce_e0.01_h0.01/checkpoint'/name)
    for mode in ('candidate_pair','group_mean_context'):
        for name,expected in full[mode]['checkpoint_sha256'].items():
            assert sha(ROOT/full[mode]['checkpoint']/name)==expected
    for name,expected in c['checkpoint_sha256'].items():assert sha(ROOT/c['checkpoint']/name)==expected
    def bscore(part,mode):return next(r for r in b[part] if r['mode']==mode)
    result=[{'name':'A_pooling_a1','stage':'A','method':'code/pooling_submission.py',
        'checkpoint':a['selected_checkpoint'],'cv_hard':[a['selection']['primary_disagreement_correct'],a['selected_confirmation_disagreement_correct']],
        'cv_all_primary':a['selection']['primary_correct'],'new_feature_count':0,'fit_and_interface_checks_pass':True},
        {'name':'B1_candidate_pair','stage':'B','method':'code/group_core.py',
        'checkpoint':full['candidate_pair']['checkpoint'],
        'cv_hard':[bscore(p,'candidate_pair')['disagreement']['correct'] for p in ('primary','confirmation')],
        'cv_all_primary':bscore('primary','candidate_pair')['all']['correct'],
        'new_feature_count':6,'fit_and_interface_checks_pass':b['all33_converged']},
        {'name':'B3_group_mean_context','stage':'B','method':'code/group_core.py',
        'checkpoint':full['group_mean_context']['checkpoint'],
        'cv_hard':[bscore(p,'group_mean_context')['disagreement']['correct'] for p in ('primary','confirmation')],
        'cv_all_primary':bscore('primary','group_mean_context')['all']['correct'],
        'new_feature_count':8,'fit_and_interface_checks_pass':b['all33_converged']},
        {'name':'C_subset','stage':'C','method':'code/group_core.py','checkpoint':c['checkpoint'],
        'cv_hard':[c[p]['disagreement']['correct'] for p in ('primary','confirmation')],
        'cv_all_primary':c['primary']['all']['correct'],'new_feature_count':8,
        'fit_and_interface_checks_pass':c['all11_converged'],
        'both_cv_above_unaugmented_B3':c['both_cv_above_unaugmented_B3']}]
    assert [str((ROOT/m['checkpoint']).resolve()) for m in result]==[
        str((ROOT/m['checkpoint']).resolve()) for m in status['pending_stage_d_validation_candidates']]
    for m in result:
        cp=ROOT/m['checkpoint'];m['asset_bytes']=assets(cp)
        m['checkpoint_sha256']={str(p.relative_to(cp)):sha(p) for p in cp.rglob('*') if p.is_file()}
    return result


def official(runner,args,output):
    global PREDICT_CALLS,SCORE_CALLS
    if runner=='predict.py':PREDICT_CALLS+=1
    else:SCORE_CALLS+=1
    command=[sys.executable,'-B',str(ROOT/'code/validation_process.py'),runner,*args]
    tick=time.perf_counter()
    p=subprocess.run(command,cwd=ROOT/'code',capture_output=True,text=True,encoding='utf-8',
                     creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    seconds=time.perf_counter()-tick
    output.with_suffix('.log').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
    if p.returncode:raise RuntimeError(f'{runner}: '+p.stderr)
    access=read(output.with_suffix('.access.json'))
    assert not access['denied'] and all(r['threads']<=4 for r in access['native_blas_threads'])
    if runner=='predict.py':
        assert all(not (Path(v).name=='gsm8k_validation.jsonl' and Path(v).parent.name!='candidates')
                   for v in access['opened'])
    return {'command':command,'wall_seconds':seconds,'returncode':p.returncode,'access':access}


def paired(new,old,hard,parsed,nc,oc):
    result={}
    for name,mask in (('all',np.ones(len(hard),dtype=bool)),('disagreement',hard),('other',~hard)):
        result[name]={'fixed':int((mask&new&~old).sum()),'broken':int((mask&~new&old).sum()),
            'net_correct':int(new[mask].sum()-old[mask].sum())}
    different=nc!=oc
    same=np.asarray([parsed[i][a] is not None and parsed[i][a]==parsed[i][b]
                     for i,(a,b) in enumerate(zip(nc,oc))])
    result.update({'different_indices':int(different.sum()),
        'different_indices_same_valid_answer':int((different&same).sum()),
        'numeric_or_invalid_answer_changes':int((different&~same).sum()),
        'invalid_selected':sum(parsed[i][j]is None for i,j in enumerate(nc))})
    return result


def execute(out):
    tick=time.perf_counter();todo=candidates();protected=snapshot()
    for name in ('common.py','predict.py','score_predictions.py'):
        assert sha(OLD/'code'/name)==sha(PROJECT/'DASE7506-MP2-main/code'/name)
    for name,entry in read(ROOT/'PLAN_MANIFEST.json')['files'].items():assert sha(ROOT/name)==entry['sha256']
    sources={str(p.relative_to(PROJECT)):sha(p) for p in
        list((ROOT/'code').glob('*.py'))+list((OLD/'code').glob('*.py'))}
    freeze={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'candidates':todo,
        'plan_sha256':sha(ROOT/'PLAN.md'),'budget_sha256':sha(ROOT/'BUDGET.json'),
        'sources_sha256':sources,'protected_before':protected,
        'new_fits':0,'max_new_validation_scoring_calls':4,'new_test_calls':0,
        'reference_cpu_eligibility':'not certified by local timings'}
    save(out/'pre_validation_freeze.json',freeze)
    raw={}
    for m in todo:
        folder=out/m['name'];folder.mkdir()
        pred,score=folder/'validation-predictions.json',folder/'validation-score.json'
        prediction=official('predict.py',['--method',str(ROOT/m['method']),
            '--checkpoint',str(ROOT/m['checkpoint']),'--split','validation','--data-dir',str(DATA),
            '--output',str(pred)],pred)
        scoring=official('score_predictions.py',['--predictions',str(pred),'--split','validation',
            '--data-dir',str(DATA),'--output',str(score)],score)
        raw[m['name']]={'prediction':read(pred),'official_score':read(score),
            'prediction_process':prediction,'scoring_process':scoring,
            'complete_validation_seconds':prediction['wall_seconds']+scoring['wall_seconds']}
        save(out/'evaluation_progress.json',{'completed':list(raw),'prediction_calls':PREDICT_CALLS,
            'official_validation_scoring_calls':SCORE_CALLS,'new_fits':0,'test_calls':0})
        print(m['name']+' '+str(raw[m['name']]['official_score']['correct'])+'/798',flush=True)
    assert PREDICT_CALLS==SCORE_CALLS==4
    bank,arrays,y=load_split('validation');hard=disagreement(bank)
    assert len(bank)==798 and int(hard.sum())==355
    parsed=[[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    baseline=read(ROOT/'runs/baseline/stage0/validation-predictions.json')
    assert baseline['ids']==[b['id'] for b in bank]
    base_choices=np.asarray(baseline['selected'],dtype=int)
    base_correct=y[np.arange(len(y)),base_choices].astype(bool)
    assert metrics(base_correct,hard)==read(ROOT/'runs/baseline/stage0/summary.json')['validation']
    choices,correctness,records={},{},{}
    for m in todo:
        name=m['name'];r=raw[name];p=r['prediction'];s=r['official_score']
        assert p['protocol']==s['protocol']==PROTOCOL and p['split']==s['split']=='validation'
        assert p['ids']==baseline['ids']
        chosen=np.asarray(p['selected'],dtype=int)
        correct=y[np.arange(len(y)),chosen].astype(bool)
        assert s['n']==798 and s['correct']==int(correct.sum())
        assert abs(s['accuracy']-100*correct.mean())<1e-12
        records[name]={**m,**metrics(correct,hard),
            'versus_ce':paired(correct,base_correct,hard,parsed,chosen,base_choices),
            'complete_validation_seconds':r['complete_validation_seconds'],
            'prediction_process':r['prediction_process'],'scoring_process':r['scoring_process']}
        choices[name],correctness[name]=chosen,correct
    b3=records['B3_group_mean_context'];c=records['C_subset']
    c['versus_unaugmented_B3']=paired(correctness['C_subset'],correctness['B3_group_mean_context'],
        hard,parsed,choices['C_subset'],choices['B3_group_mean_context'])
    eligible=[]
    for m in todo:
        name=m['name'];r=records[name]
        tests={'both_training_cv_strictly_above_ce':r['cv_hard'][0]>1743 and r['cv_hard'][1]>1749,
            'validation_disagreement_at_least309':r['disagreement']['correct']>=309,
            'validation_overall_at_least748':r['all']['correct']>=748,
            'frozen_source_fit_and_interface_checks':r['fit_and_interface_checks_pass'],
            'local_resource_screen':r['asset_bytes']<=16*2**20 and r['complete_validation_seconds']<=30}
        if name=='C_subset':
            tests['augmentation_above_B3_on_both_cv']=r['both_cv_above_unaugmented_B3']
            tests['validation_not_below_unaugmented_B3']=r['all']['correct']>=b3['all']['correct'] and r['disagreement']['correct']>=b3['disagreement']['correct']
        r['eligibility_checks']=tests;r['passes_predeclared_gate']=all(tests.values())
        r['failed_checks']=[k for k,v in tests.items() if not v]
        if r['passes_predeclared_gate']:eligible.append(r)
    def priority(r):
        return (-r['cv_hard'][0],-min(r['cv_hard'][0]-1743,r['cv_hard'][1]-1749),
                -r['cv_all_primary'],r['asset_bytes'],1,r['new_feature_count'],r['name'])
    eligible.sort(key=priority)
    selected=eligible[0]['name'] if eligible else None
    assert {p:sha(p) for p in protected}==protected
    assert {p:sha(PROJECT/p) for p in sources}==sources
    assert not DENIED
    decision={'status':'eligible_candidate_frozen_for_later_test' if selected else 'completed_no_qualified_candidate',
        'selected_candidate':selected,'all_candidates_pass_fail':{n:r['passes_predeclared_gate'] for n,r in records.items()},
        'failed_checks':{n:r['failed_checks'] for n,r in records.items()},
        'selection_rule':'Primary TRAIN CV, worst CV gain, primary overall, assets, members, new features, name; validation is gate only.',
        'new_training_fits':0,'new_official_validation_scoring_calls':4,'new_test_calls':0,
        'retained_formal_submission':'frozen-v1.1 unchanged','no_parameter_changes_after_validation':True,
        'no_automatic_test_or_publication':True,'no_automatic_additional_training':True,
        'reference_cpu_eligibility':'Local validation screening only; reference Xeon qualification remains unmeasured.'}
    save(out/'DECISION.json',decision)
    result={'stage':'D','status':'completed','finished_utc':datetime.now(timezone.utc).isoformat(),
        'ce_baseline':metrics(base_correct,hard),'records':records,'decision':decision,
        'new_training_fits':0,'new_prediction_calls':4,'new_official_validation_scoring_calls':4,
        'new_test_calls':0,'parent_data_accesses':sorted(OPENED),'denied_accesses':DENIED,
        'protected_artifacts_verified_unchanged':len(protected),'frozen_sources_unchanged':True,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'threads':4,'workers':1,
            'cpu':read(ROOT/'runs/baseline/stage0/summary.json')['environment']['cpu'],
            'reference_cpu_measured':False},'wall_seconds':time.perf_counter()-tick,
        'limits':'Repeated development validation; no unbiased or multiple-comparison-corrected inference.'}
    save(out/'summary.json',result)
    print({'stage_d':'completed','records':{n:[r['all']['correct'],r['disagreement']['correct']]for n,r in records.items()},
        'selected_candidate':selected,'test_calls':0},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/validation/stage_d')
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT) or out.exists():raise ValueError('Use a fresh V6 output directory')
    out.mkdir(parents=True)
    try:execute(out)
    except Exception as error:
        save(out/'failure.json',{'type':type(error).__name__,'message':str(error),
            'prediction_calls':PREDICT_CALLS,'scoring_calls':SCORE_CALLS,'new_fits':0,'test_calls':0,'denied':DENIED})
        raise


if __name__=='__main__':main()
