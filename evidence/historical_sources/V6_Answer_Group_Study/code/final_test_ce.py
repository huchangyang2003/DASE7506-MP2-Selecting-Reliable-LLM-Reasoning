"""One user-requested frozen CE test, with saved A as the pooling comparison."""
# Reuse only receipt/hash helpers and the parent isolation guard, never A.execute().
import final_test_a as base
import os
import platform
import subprocess
import sys
import time
import zipfile
import numpy as np

ROOT,PROJECT,OFFICIAL,DATA=base.ROOT,base.PROJECT,base.OFFICIAL,base.DATA
read,sha,save,utc=base.read,base.sha,base.save,base.utc
OUT=ROOT/'runs/final_test/ce_baseline_20261030'
CP=PROJECT/'V3_Block_Regularization/runs/step1/uniform_ce_e0.01_h0.01/checkpoint'
METHOD=PROJECT/'V3_Block_Regularization/code/submission.py'
AUTH=ROOT/'CE_FINAL_TEST_AUTHORIZATION_20261030.json'
A_RUN=ROOT/'runs/final_test/a_pooling_a1_20261030'

def execute():
    if OUT.exists():raise FileExistsError('Never overwrite or repeat this CE test')
    auth=read(AUTH);status=read(ROOT/'STATUS.json')
    assert auth['user_explicitly_requested_ce_test'] and auth['candidate']=='CE_uniform_e0.01_h0.01'
    assert auth['max_official_test_prediction_calls']==auth['max_official_test_scoring_calls']==1
    assert status['official_test_calls']==1 and not status.get('ce_final_test_completed',False)
    a_freeze=read(A_RUN/'PRE_TEST_FREEZE.json');a_summary=read(A_RUN/'summary.json')
    assert a_summary['official_test']['correct']==393 and a_summary['audit']['passed']
    for p,digest in a_freeze['sources_sha256'].items():assert sha(p)==digest
    cp_hash={str(p.relative_to(CP)):sha(p) for p in CP.rglob('*') if p.is_file()}
    a_base_hash={name.replace('\\','/').removeprefix('baseline/'):digest for name,digest in
        a_freeze['candidate']['checkpoint_sha256'].items() if name.replace('\\','/').startswith('baseline/')}
    assert cp_hash==a_base_hash
    protected=base.protect()
    sources={str(p):sha(p) for folder in (ROOT/'code',PROJECT/'V3_Block_Regularization/code',OFFICIAL)
        for p in folder.glob('*.py')}
    manifest=read(DATA/'manifest.json')
    assert manifest['protocol']==base.PROTOCOL and manifest['splits']['test']==467
    OUT.mkdir(parents=True)
    for name in ('STATUS.json','DECISION.json','ARTIFACT_MANIFEST.json','RESULTS.md','README.md','FINAL_TEST_RESULTS.md'):
        (OUT/('previous_'+name)).write_bytes((ROOT/name).read_bytes())
    freeze={'frozen_at_utc':utc(),'candidate':auth['candidate'],'method':str(METHOD),
        'checkpoint':str(CP),'checkpoint_sha256':cp_hash,'identical_to_a_base_checkpoint':True,
        'authorization_sha256':sha(AUTH),'sources_sha256':sources,'protected_before':protected,
        'data_manifest_sha256':sha(DATA/'manifest.json'),
        'expected_test_data_sha256':{k:v for k,v in manifest['sha256'].items() if 'gsm8k_test.' in k},
        'reused_a_predictions_sha256':sha(A_RUN/'test-predictions.json'),
        'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
        'processor_identifier':os.environ.get('PROCESSOR_IDENTIFIER'),
        'max_test_prediction_calls':1,'max_official_test_scoring_calls':1,'new_fits':0,
        'purpose':'Post-A-test supplementary unpooled control, not new parameter/model search',
        'first_blind_test':False,'a_and_historical_test_results_already_known':True,
        'publication_automatic':False,
        'execution_notes':['An initial preflight stopped before creating this output or calling any runner: Windows backslashes in A manifest keys were normalized for the baseline hash comparison. No inference, scoring, training, model or configuration change occurred.']}
    save(OUT/'PRE_TEST_FREEZE.json',freeze)
    ledger={'started_at_utc':utc(),'prediction_calls':0,'official_test_scoring_calls':0,
        'new_fits':0,'a_prediction_calls':0}
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
    pred_path=OUT/'test-predictions.json';score_path=OUT/'test-score.json'
    prediction=official('predict.py',['--method',str(METHOD),'--checkpoint',str(CP),
        '--split','test','--data-dir',str(DATA),'--output',str(pred_path)],pred_path)
    pred_hash=sha(pred_path)
    save(OUT/'PREDICTIONS_FROZEN_BEFORE_SCORING.json',{'sha256':pred_hash,'frozen_at_utc':utc()})
    scoring=official('score_predictions.py',['--predictions',str(pred_path),'--split','test',
        '--data-dir',str(DATA),'--output',str(score_path)],score_path)
    base.AFTER_SCORE=True
    pred=read(pred_path);result=read(score_path)
    assert pred['protocol']==result['protocol']==base.PROTOCOL and pred['split']==result['split']=='test'
    assert sha(pred_path)==pred_hash
    bank=base.read_jsonl(DATA/'candidates/gsm8k_test.jsonl');refs=base.read_jsonl(DATA/'gsm8k_test.jsonl')
    assert len(bank)==len(refs)==467 and pred['ids']==[r['id'] for r in bank]
    assert all(b['id']==r['id'] and b['question']==r['question'] for b,r in zip(bank,refs))
    parsed=[[base.parse_answer(c['text']) for c in row['candidates']] for row in bank]
    y=np.asarray([[a is not None and a==base.parse_answer(ref['answer']) for a in row] for row,ref in zip(parsed,refs)])
    hard=np.asarray([len({a for a in row if a is not None})>=2 for row in parsed]);assert hard.all()
    ce_choices=np.asarray(pred['selected'],dtype=int);ce=y[np.arange(467),ce_choices]
    assert result['correct']==int(ce.sum()) and result['n']==467
    assert abs(result['accuracy']-100*ce.mean())<1e-12
    a_pred=read(A_RUN/'test-predictions.json')
    formal_pred=read(PROJECT/'Final_Optimized_Solution/evaluation/final_test/test-predictions.json')
    assert a_pred['ids']==formal_pred['ids']==pred['ids']
    a_choices=np.asarray(a_pred['selected'],dtype=int);formal_choices=np.asarray(formal_pred['selected'],dtype=int)
    a=y[np.arange(467),a_choices];formal=y[np.arange(467),formal_choices]
    assert int(a.sum())==393 and int(formal.sum())==384
    def pair(new,old,nc,oc):
        changed=nc!=oc
        same=np.asarray([parsed[i][n] is not None and parsed[i][n]==parsed[i][o]
            for i,(n,o) in enumerate(zip(nc,oc))])
        return {'n':467,'new_correct':int(new.sum()),'control_correct':int(old.sum()),
            'fixed':int((new&~old).sum()),'broken':int((~new&old).sum()),
            'net_correct':int(new.sum()-old.sum()),'different_indices':int(changed.sum()),
            'different_indices_same_valid_answer':int((changed&same).sum()),
            'numeric_or_invalid_answer_changes':int((changed&~same).sum())}
    paired={'A_vs_CE':pair(a,ce,a_choices,ce_choices),
        'CE_vs_historical_formal':pair(ce,formal,ce_choices,formal_choices)}
    assets=0
    for p in CP.rglob('*'):
        if p.is_file():
            if p.suffix=='.npz':
                with zipfile.ZipFile(p) as z:assets+=sum(i.file_size for i in z.infolist())
            else:assets+=p.stat().st_size
    gold=str((DATA/'gsm8k_test.jsonl').resolve())
    assert gold not in prediction['access']['opened'] and gold in scoring['access']['opened']
    assert {p:sha(p) for p in protected}==protected and {p:sha(p) for p in sources}==sources
    assert {str(p.relative_to(CP)):sha(p) for p in CP.rglob('*') if p.is_file()}==cp_hash
    assert sha(AUTH)==freeze['authorization_sha256'] and sha(DATA/'manifest.json')==freeze['data_manifest_sha256']
    assert sha(A_RUN/'test-predictions.json')==freeze['reused_a_predictions_sha256']
    seconds=prediction['wall_seconds']+scoring['wall_seconds']
    summary={'completed_at_utc':utc(),'candidate':auth['candidate'],'official_test':result,
        'paired':paired,'prediction_process':prediction,'scoring_process':scoring,
        'total_fresh_prediction_and_scoring_seconds':seconds,'uncompressed_inference_asset_bytes':assets,
        'local_resource_screen_passed':seconds<=30 and assets<=16*2**20,'reference_cpu_certified':False,
        'audit':{'passed':True,'official_score_reconciled':True,'prediction_never_opened_gold':True,
        'predictions_frozen_before_gold_scoring':True,'prior_artifacts_preserved':len(protected),
        'source_files_unchanged':len(sources),'checkpoint_unchanged':True,'a_checkpoint_base_byte_identical':True},
        'new_real_fits':0,'new_validation_calls':0,'new_test_prediction_calls':1,
        'new_official_test_scoring_calls':1,'a_rerun':False,'first_blind_test':False,
        'formal_submission_changed':False,'no_followup_tuning':True}
    save(OUT/'summary.json',summary)
    save(OUT/'COMPLETED.json',{**ledger,'completed_at_utc':utc(),'summary_sha256':sha(OUT/'summary.json')})
    print(base.json.dumps({'official_test':result,'paired':paired,'resource_seconds':seconds,
        'asset_bytes':assets,'audit':summary['audit']},ensure_ascii=False))

if __name__=='__main__':execute()
