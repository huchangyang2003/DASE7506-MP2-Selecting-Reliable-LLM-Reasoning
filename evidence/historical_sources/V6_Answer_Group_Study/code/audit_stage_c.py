"""Rebuild all stage-C subset pairs and verify the 11 saved fits without optimization."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
from copy import deepcopy
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
import stage_c as sc
from audit_stage_b import independent_feature,decode
from group_core import build_inputs,group_values,load_model,model_scores,choose_scores,predict
from experiment_io import load_split

CONTRACT=False
ALLOWED,READS=set(),set()


def guard(event,args):
    if CONTRACT and event=='open' and isinstance(args[0],(str,bytes)):
        path=str(Path(os.fsdecode(args[0])).resolve());READS.add(path)
        if path not in ALLOWED:raise PermissionError('Inference may read only its checkpoint')


sys.addaudithook(guard)


def audit_fit(folder,bank,x,parsed,groups,y,fit_rows):
    info=sc.sb.io.read(folder/'statistics_provenance.json')
    source=sc.ROOT/info['source']
    assert sc.sb.io.sha(source)==info['source_sha256']==sc.sb.io.sha(folder/'statistics_from_B.npz')
    with np.load(source,allow_pickle=False) as z:
        stats={k:z[k].copy() for k in z.files}
    assert np.array_equal(stats['train_rows'],fit_rows)
    model=load_model(folder/'checkpoint')
    assert model['mode']==sc.MODE
    assert np.array_equal(model['mean'],stats['mean']) and np.array_equal(model['scale'],stats['scale'])
    with np.load(folder/'pair_audit.npz',allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files}
    expected={};expected_informative=[]
    for q in fit_rows:
        good=[m for _,m in groups[q] if y[q,m[0]]]
        bad=[m for _,m in groups[q] if not y[q,m[0]]]
        if not good or not bad:continue
        expected_informative.append(q)
        for p in good:
            for n in bad:
                fp=sum(1<<j for j in p);fn=sum(1<<j for j in n)
                expected[(q,fp,fn)]={'pairs':(2**len(p)-1)*(2**len(n)-1),
                    'row_weight':1/(len(good)*len(bad)*(2**len(p)-1)*(2**len(n)-1)),
                    'total_weight':1/(len(good)*len(bad))}
    observed={};X=[];expected_weights=[]
    for q,pm,nm,fp,fn,w in zip(a['owners'],a['positive_mask'],a['negative_mask'],
                              a['full_positive_mask'],a['full_negative_mask'],a['row_weights']):
        key=(int(q),int(fp),int(fn));plan=expected[key]
        assert pm and nm and not (int(pm)&int(nm))
        assert int(pm)&int(fp)==int(pm) and int(nm)&int(fn)==int(nm)
        p,n=decode(pm),decode(nm)
        assert y[q,p].all() and not y[q,n].any()
        assert len({parsed[q][j] for j in p})==len({parsed[q][j] for j in n})==1
        assert all(parsed[q][j] is not None for j in p+n)
        assert abs(w-plan['row_weight'])<1e-15
        item=observed.setdefault(key,{'masks':set(),'weight':0.})
        pair=(int(pm),int(nm));assert pair not in item['masks']
        item['masks'].add(pair);item['weight']+=w
        X.append((independent_feature(x,bank,q,p,sc.MODE)-
                  independent_feature(x,bank,q,n,sc.MODE))/model['scale'])
        expected_weights.append(plan['row_weight'])
    assert set(observed)==set(expected)
    for key,plan in expected.items():
        item=observed[key]
        assert len(item['masks'])==plan['pairs'] and abs(item['weight']-plan['total_weight'])<1e-12
    assert np.array_equal(a['informative_questions'],expected_informative)
    totals=np.bincount(a['owners'],weights=a['row_weights'],minlength=len(bank))
    assert np.allclose(totals[expected_informative],1,rtol=1e-13,atol=1e-13)
    assert set(a['owners'])<=set(fit_rows)
    X=np.asarray(X);weights=np.asarray(expected_weights);norm=weights/weights.sum()
    margin=X@model['weights']
    value=float(norm@np.logaddexp(0,-margin)+.005*(model['weights']@model['weights']))
    grad=-(X.T@(norm*np.exp(-np.logaddexp(0,margin))))+.01*model['weights']
    opt=sc.sb.io.read(folder/'optimization.json')
    assert abs(value-opt['final_loss'])<1e-12 and np.abs(grad).max()<=1e-6
    assert abs(float(np.abs(grad).max())-opt['gradient_inf'])<1e-12
    return model,{'folder':str(folder.relative_to(sc.ROOT)),'subset_pair_rows':len(X),
        'original_answer_group_pairs':len(expected),'informative_questions':len(expected_informative),
        'all_nonempty_subset_pairs_complete_and_unique':True,'all_group_and_question_weights_verified':True,
        'statistics_byte_identical_to_B':True,'gradient_inf':float(np.abs(grad).max())}


def main():
    global CONTRACT,ALLOWED
    start=time.perf_counter();run=sc.ROOT/'runs/subset_augmentation/stage_c';output=run/'audit.json'
    if output.exists():raise FileExistsError(output)
    summary=sc.sb.io.read(run/'summary.json');initial=sc.sb.io.read(run/'started.json')
    assert summary['status']=='completed' and summary['new_real_fits']==11
    assert {p:sc.sb.io.sha(p) for p in initial['protected_before']}==initial['protected_before']
    assert {n:sc.sb.io.sha(sc.ROOT/'code'/n) for n in initial['sources_sha256']}==initial['sources_sha256']
    assert sc.sb.io.sha(sc.B_RUN/'selection_before_confirmation.json')==initial['fixed_B_selection_sha256']
    bank,arrays,y=load_split('train');x,parsed,groups=build_inputs(bank,arrays);gv=group_values(x,groups)
    records=[]
    for partition in ('primary','confirmation','full'):
        if partition=='full':iterations=[None]
        else:
            with np.load(run/partition/'oof_scores.npz',allow_pickle=False) as z:
                saved={k:z[k].copy() for k in z.files}
            with np.load(sc.B_RUN/partition/'oof_scores.npz',allow_pickle=False) as z:
                assert np.array_equal(saved['folds'],z['folds']) and np.array_equal(saved['labels'],z['labels'])
            iterations=range(5)
        for f in iterations:
            fit_rows=np.arange(len(bank)) if f is None else np.flatnonzero(saved['folds']!=f)
            folder=run/'full' if f is None else run/partition/f'fold_{f}'
            model,record=audit_fit(folder,bank,x,parsed,groups,y,fit_rows)
            if f is not None:
                held=np.flatnonzero(saved['folds']==f)
                hb=[bank[i] for i in held];hp=[parsed[i] for i in held]
                hg=[groups[i] for i in held];hv=[gv[i] for i in held]
                scores=model_scores(hb,x[held],hp,hg,hv,model)
                assert np.array_equal(scores,saved['scores'][held])
                assert np.array_equal(choose_scores(hb,x[held],hp,scores),saved['selected'][held])
                record['oof_scores_and_choices_exact']=True
            records.append(record)
    cp=run/'full/checkpoint';ALLOWED={str(p.resolve()) for p in cp.iterdir()}
    small=bank[:32];aa={k:v[:32] for k,v in arrays.items()}
    CONTRACT=True
    selected=predict(small,aa,cp)
    assert all(type(c)is int and 0<=c<4 for c in selected)
    for n in (1,3,7,32):assert predict(small[:n],{k:v[:n] for k,v in aa.items()},cp)==selected[:n]
    assert predict([],{k:v[:0] for k,v in aa.items()},cp)==[]
    changed=deepcopy(small)
    for i,row in enumerate(changed):row['id']=f'c-new-{i}'
    assert predict(changed,aa,cp)==selected
    perm=np.array([2,0,3,1])
    for row in changed:
        row['candidates']=[{**row['candidates'][j],'index':k} for k,j in enumerate(perm)]
    pp=predict(changed,{k:v[:,perm] for k,v in aa.items()},cp)
    rp=sc.sb.io.answers(changed)
    assert [parsed[i][j] for i,j in enumerate(selected)]==[rp[i][j] for i,j in enumerate(pp)]
    unknown=deepcopy(small[:1])
    for c in unknown[0]['candidates']:c['text']='Unknown answer'
    assert predict(unknown,{k:v[:1] for k,v in aa.items()},cp)==[0]
    CONTRACT=False
    with zipfile.ZipFile(cp/'model.npz')as z:assets=sum(i.file_size for i in z.infolist())
    assets+=(cp/'config.json').stat().st_size
    assert assets<16*2**20 and len(records)==11 and not sc.sb.io.DENIED
    assert {p:sc.sb.io.sha(p) for p in initial['protected_before']}==initial['protected_before']
    sc.sb.io.save(output,{'passed':True,'reconstructed_fits':records,'statistics_reestimated':0,
        'statistics_byte_reused_from_B':11,'exact_oof_models_reproduced':10,
        'inference_contracts':{'train_questions':32,'batches':[1,3,7,32],'empty':True,
            'new_ids':True,'candidate_answer_permutation':True,'all_invalid_fallback':True},
        'inference_checkpoint_reads':sorted(READS),'uncompressed_asset_bytes':assets,
        'new_optimizer_calls':0,'validation_calls':0,'test_calls':0,
        'native_blas_threads':sc.sb.blas_threads(),'seconds':time.perf_counter()-start,
        'bank_accesses':sorted(p for p in sc.sb.io.OPENED if Path(p).name.startswith('gsm8k_'))})
    print({'stage_c_audit':'passed','fits_reconstructed':11,'B_statistics_reused':11,
        'OOF_models_exact':10,'asset_bytes':assets},flush=True)


if __name__=='__main__':main()
