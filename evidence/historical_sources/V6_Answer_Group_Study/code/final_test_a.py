"""User-authorized one-off test of the unchanged A checkpoint; no fitting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
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

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
OFFICIAL=PROJECT/'DASE7506-MP2-main/code'
DATA=OFFICIAL/'data'
OUT=ROOT/'runs/final_test/a_pooling_a1_20261030'
CP=ROOT/'runs/pooling/stage_a/selected_checkpoint'
METHOD=ROOT/'code/pooling_submission.py'
AUTH=ROOT/'FINAL_TEST_AUTHORIZATION_20261030.json'
sys.path.insert(0,str(OFFICIAL))
from common import PROTOCOL,parse_answer,read_jsonl

AFTER_SCORE=False
def parent_guard(event,args):
    if event.startswith('socket.connect'):raise PermissionError('Offline evaluation')
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=Path(os.fsdecode(args[0])).resolve()
        if p.name.lower().startswith('gsm8k_'):
            allowed={(DATA/'gsm8k_test.jsonl').resolve(),(DATA/'candidates/gsm8k_test.jsonl').resolve()}
            if not AFTER_SCORE or p not in allowed:raise PermissionError('Parent may read test text only after scoring')
sys.addaudithook(parent_guard)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def utc():return datetime.now(timezone.utc).isoformat()

def protect():
    result={}
    for base in (ROOT,PROJECT/'V3_Block_Regularization'):
        manifest=base/'ARTIFACT_MANIFEST.json';result[str(manifest)]=sha(manifest)
        for relative,entry in read(manifest)['files'].items():
            p=base/relative;assert sha(p)==entry['sha256'],str(p);result[str(p)]=entry['sha256']
    frozen=PROJECT/'Final_Optimized_Solution'
    for relative,entry in read(frozen/'FREEZE.json')['files'].items():
        p=frozen/relative;assert sha(p)==entry['sha256'];result[str(p)]=entry['sha256']
    for p in (frozen/'FREEZE.json',frozen/'output/pdf/DASE7506_MP2_Final_Report.pdf'):
        result[str(p)]=sha(p)
    pub=PROJECT/'First_Publication_20261022/DASE7506-MP2-Selecting-Reliable-LLM-Reasoning'
    result[str(pub/'PUBLICATION_MANIFEST.json')]=sha(pub/'PUBLICATION_MANIFEST.json')
    for relative in ('checkpoint/model.npz','checkpoint/config.json','REPORT.pdf'):
        p=pub/relative;assert sha(p)==read(pub/'PUBLICATION_MANIFEST.json')['files'][relative]['sha256']
        result[str(p)]=sha(p)
    return result

def execute():
    global AFTER_SCORE
    if OUT.exists():raise FileExistsError('Do not repeat or overwrite the final test')
    auth=read(AUTH)
    assert auth['candidate']=='A_pooling_a1' and auth['max_official_test_scoring_calls']==1
    assert auth['user_explicitly_authorized'] and auth['no_parameter_changes']
    status=read(ROOT/'STATUS.json');assert status['official_test_calls']==0
    config=read(CP/'config.json');assert config['alpha']==1 and config['temperature']==1
    prior=read(ROOT/'runs/validation/stage_d/pre_validation_freeze.json')
    candidate=next(m for m in prior['candidates'] if m['name']=='A_pooling_a1')
    assert {str(p.relative_to(CP)):sha(p) for p in CP.rglob('*') if p.is_file()}==candidate['checkpoint_sha256']
    for relative,digest in prior['sources_sha256'].items():assert sha(PROJECT/relative)==digest
    protected=protect()
    sources={str(p):sha(p) for folder in (ROOT/'code',PROJECT/'V3_Block_Regularization/code',OFFICIAL)
             for p in folder.glob('*.py')}
    manifest=read(DATA/'manifest.json')
    assert manifest['protocol']==PROTOCOL and manifest['splits']['test']==467
    OUT.mkdir(parents=True)
    for name in ('STATUS.json','DECISION.json','ARTIFACT_MANIFEST.json','RESULTS.md','README.md'):
        (OUT/('previous_'+name)).write_bytes((ROOT/name).read_bytes())
    freeze={'frozen_at_utc':utc(),'candidate':candidate,'authorization_sha256':sha(AUTH),
        'sources_sha256':sources,'protected_before':protected,'data_manifest_sha256':sha(DATA/'manifest.json'),
        'expected_test_data_sha256':{k:v for k,v in manifest['sha256'].items() if 'gsm8k_test.' in k},
        'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
        'processor_identifier':os.environ.get('PROCESSOR_IDENTIFIER'),
        'max_test_prediction_calls':1,'max_official_test_scoring_calls':1,'new_fits':0,
        'development_and_historical_test_results_previously_known':True,
        'is_first_blind_test':False,'post_validation_gate_exception':True,
        'no_tuning_or_other_research_candidate_selection_after_test':True,
        'publication_automatic':False,'reference_cpu_certified':False}
    save(OUT/'PRE_TEST_FREEZE.json',freeze)
    ledger={'started_at_utc':utc(),'prediction_calls':0,'official_test_scoring_calls':0,'new_fits':0}
    save(OUT/'STARTED.json',ledger)
    def official(runner,args,output):
        key='prediction_calls' if runner=='predict.py' else 'official_test_scoring_calls'
        assert ledger[key]==0
        ledger[key]+=1;save(OUT/'CALL_LEDGER.json',ledger)
        command=[sys.executable,'-B',str(ROOT/'code/final_test_process.py'),runner,*args]
        tick=time.perf_counter()
        p=subprocess.run(command,cwd=ROOT/'code',capture_output=True,text=True,encoding='utf-8',
            timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        elapsed=time.perf_counter()-tick
        output.with_suffix('.log').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
        if p.returncode:raise RuntimeError(p.stderr)
        access=read(output.with_suffix('.access.json'))
        assert not access['denied'] and all(r['threads']<=4 for r in access['native_blas_threads'])
        return {'command':command,'wall_seconds':elapsed,'returncode':p.returncode,'access':access}
    predictions=OUT/'test-predictions.json';score_path=OUT/'test-score.json'
    prediction=official('predict.py',['--method',str(METHOD),'--checkpoint',str(CP),
        '--split','test','--data-dir',str(DATA),'--output',str(predictions)],predictions)
    pred_hash=sha(predictions)
    save(OUT/'PREDICTIONS_FROZEN_BEFORE_SCORING.json',{'sha256':pred_hash,'frozen_at_utc':utc()})
    scoring=official('score_predictions.py',['--predictions',str(predictions),'--split','test',
        '--data-dir',str(DATA),'--output',str(score_path)],score_path)
    AFTER_SCORE=True
    result=read(score_path);pred=read(predictions)
    assert result['protocol']==pred['protocol']==PROTOCOL and result['split']==pred['split']=='test'
    assert sha(predictions)==pred_hash
    bank=read_jsonl(DATA/'candidates/gsm8k_test.jsonl');refs=read_jsonl(DATA/'gsm8k_test.jsonl')
    assert len(bank)==len(refs)==467 and pred['ids']==[r['id'] for r in bank]
    assert all(b['id']==r['id'] and b['question']==r['question'] for b,r in zip(bank,refs))
    parsed=[[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    y=np.asarray([[a is not None and a==parse_answer(ref['answer']) for a in row] for row,ref in zip(parsed,refs)])
    hard=np.asarray([len({a for a in row if a is not None})>=2 for row in parsed])
    choices=np.asarray(pred['selected'],dtype=int);correct=y[np.arange(len(bank)),choices]
    assert result['correct']==int(correct.sum()) and result['n']==len(bank)
    assert abs(result['accuracy']-100*correct.mean())<1e-12
    original=read(PROJECT/'Final_Optimized_Solution/evaluation/final_test/test-predictions.json')
    original_score=read(PROJECT/'Final_Optimized_Solution/evaluation/final_test/test-score.json')
    assert original['protocol']==PROTOCOL and original['split']=='test' and original['ids']==pred['ids']
    old_choices=np.asarray(original['selected'],dtype=int);old_correct=y[np.arange(len(bank)),old_choices]
    assert original_score['correct']==int(old_correct.sum())==384
    paired={}
    for name,mask in (('all',np.ones(len(bank),dtype=bool)),('disagreement',hard),('other',~hard)):
        paired[name]={'correct':int(correct[mask].sum()),'n':int(mask.sum()),
            'historical_formal_correct':int(old_correct[mask].sum()),
            'fixed':int((mask&correct&~old_correct).sum()),'broken':int((mask&~correct&old_correct).sum()),
            'net_correct':int(correct[mask].sum()-old_correct[mask].sum())}
    assets=0
    for p in CP.rglob('*'):
        if p.is_file():
            if p.suffix=='.npz':
                with zipfile.ZipFile(p) as z:assets+=sum(i.file_size for i in z.infolist())
            else:assets+=p.stat().st_size
    assert {p:sha(p) for p in protected}==protected
    assert {p:sha(p) for p in sources}==sources
    assert sha(AUTH)==freeze['authorization_sha256'] and sha(DATA/'manifest.json')==freeze['data_manifest_sha256']
    assert {str(p.relative_to(CP)):sha(p) for p in CP.rglob('*') if p.is_file()}==candidate['checkpoint_sha256']
    gold=str((DATA/'gsm8k_test.jsonl').resolve())
    assert gold not in prediction['access']['opened'] and gold in scoring['access']['opened']
    seconds=prediction['wall_seconds']+scoring['wall_seconds']
    summary={'completed_at_utc':utc(),'candidate':'A_pooling_a1','official_test':result,
        'versus_historical_formal':paired,'prediction_process':prediction,'scoring_process':scoring,
        'total_fresh_prediction_and_scoring_seconds':seconds,'uncompressed_inference_asset_bytes':assets,
        'local_resource_screen_passed':seconds<=30 and assets<=16*2**20,
        'reference_cpu_certified':False,'audit':{'passed':True,'official_score_reconciled':True,
        'predictions_frozen_before_gold_scoring':True,'prediction_never_opened_gold':True,
        'prior_artifacts_preserved':len(protected),'source_files_unchanged':len(sources),
        'checkpoint_unchanged':True},'new_real_fits':0,'new_validation_calls':0,
        'new_test_prediction_calls':ledger['prediction_calls'],'new_official_test_scoring_calls':ledger['official_test_scoring_calls'],
        'post_validation_user_authorized_exception':True,'first_blind_test':False,
        'formal_submission_changed':False,'no_followup_tuning':True}
    save(OUT/'summary.json',summary)
    save(OUT/'COMPLETED.json',{'completed_at_utc':utc(),**ledger,'summary_sha256':sha(OUT/'summary.json')})
    print(json.dumps({'official_test':result,'versus_historical_formal':paired,
        'resource_seconds':seconds,'asset_bytes':assets,'audit':summary['audit']},ensure_ascii=False))

if __name__=='__main__':execute()
