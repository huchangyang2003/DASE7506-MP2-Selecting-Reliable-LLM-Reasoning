"""Rebuild fixed TRAIN-only CE; optionally reproduce both CV pooling comparisons."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from common import load_inputs,read_jsonl,parse_answer
from linear_core import build_inputs,fit_scaler,fit
from pooling_core import choose,ALPHAS
from fusion_core import grouped_folds

def save(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--destination',type=Path,required=True)
    ap.add_argument('--data-dir',type=Path,default=ROOT/'data')
    ap.add_argument('--with-cv',action='store_true',help='10 more fixed CE fits; all four predeclared alphas, no new search')
    args=ap.parse_args();data=args.data_dir.resolve();dest=args.destination.resolve()
    if dest.exists():raise FileExistsError('Use a new destination; never overwrite evidence')
    dest.mkdir(parents=True)
    opened=set();denied=[];calls=0;tick=time.perf_counter()
    allowed={(data/'gsm8k_train.jsonl').resolve(),(data/'candidates/gsm8k_train.jsonl').resolve(),(data/'candidates/gsm8k_train.npz').resolve()}
    def audit(event,values):
        if event.startswith('socket.connect'):raise PermissionError('Training reproduction is offline')
        if event=='open' and isinstance(values[0],(str,bytes)):
            p=Path(os.fsdecode(values[0])).resolve()
            if p.name.lower().startswith('gsm8k_'):
                if p not in allowed:denied.append(str(p));raise PermissionError('Training may read TRAIN only')
                opened.add(str(p))
    sys.addaudithook(audit)
    bank,arrays=load_inputs('train',data);refs=read_jsonl(data/'gsm8k_train.jsonl')
    assert len(bank)==len(refs) and all(b['id']==r['id'] and b['question']==r['question'] for b,r in zip(bank,refs))
    y=np.asarray([[int((a:=parse_answer(c['text'])) is not None and a==parse_answer(r['answer'])) for c in b['candidates']] for b,r in zip(bank,refs)],float)
    hard=np.asarray([len({a for c in b['candidates'] if (a:=parse_answer(c['text'])) is not None})>=2 for b in bank])
    x,names=build_inputs(bank,arrays)
    assert x.shape==(4755,4,1071) and int(hard.sum())==2114
    def fitting(xx,yy,label):
        nonlocal calls
        calls+=1;save(dest/'CALL_LEDGER.json',{'fit_calls':calls,'current':label,'with_cv':args.with_cv,'validation_calls':0,'test_calls':0})
        theta,record=fit(xx,yy,'uniform_ce',0.01,0.01)
        save(dest/'optimization'/f'{label}.json',record)
        if not record['converged'] or record['gradient_inf']>1e-6:raise RuntimeError('Fixed fit did not converge')
        return theta
    cv=[]
    if args.with_cv:
        for seed in (20261022,20261029):
            folds=grouped_folds(bank,seed=seed);oof=np.full(y.shape,np.nan)
            for f in range(5):
                train,held=np.flatnonzero(folds!=f),np.flatnonzero(folds==f)
                mean,scale=fit_scaler(x[train]);theta=fitting(np.ascontiguousarray((x[train]-mean)/scale),y[train],f's{seed}_fold{f}')
                oof[held]=((x[held]-mean)/scale)@theta[:-1]+theta[-1]
            scores=[]
            for alpha in ALPHAS:
                selected=np.asarray(choose(bank,oof,alpha,arrays['embeddings'])[0]);ok=y[np.arange(len(bank)),selected].astype(bool)
                scores.append({'alpha':alpha,'all_correct':int(ok.sum()),'disagreement_correct':int(ok[hard].sum())})
            cv.append({'seed':seed,'scores':scores})
            np.savez(dest/f'oof_{seed}.npz',logits=oof,fold=folds)
        selected=max(cv[0]['scores'],key=lambda s:(s['disagreement_correct'],s['all_correct'],-s['alpha']))['alpha']
        assert selected==1.0,'Different reconstructed OOF selection; inspect without retuning'
        save(dest/'pooling_cv.json',{'cv':cv,'selected_alpha_from_primary_only':selected})
    mean,scale=fit_scaler(x);theta=fitting(np.ascontiguousarray((x-mean)/scale),y,'full_train')
    cp=dest/'checkpoint';(cp/'baseline').mkdir(parents=True)
    np.savez(cp/'baseline/model.npz',mean=mean,scale=scale,weights=theta[:-1],bias=np.asarray(theta[-1]))
    for rel in ('config.json','baseline/config.json'):shutil.copyfile(ROOT/'checkpoint'/rel,cp/rel)
    assert names==json.loads((cp/'baseline/config.json').read_text())['feature_names']
    with np.load(ROOT/'checkpoint/baseline/model.npz',allow_pickle=False) as original:
        differences={k:float(np.max(np.abs(original[k]-v))) for k,v in {'mean':mean,'scale':scale,'weights':theta[:-1],'bias':np.asarray(theta[-1])}.items()}
    passed=all(d<=1e-10 for d in differences.values())
    record={'passed':passed,'purpose':'Fixed training reproduction only; not method selection','fit_calls':calls,'with_cv':args.with_cv,
        'validation_calls':0,'test_calls':0,'all_training_bank_files':sorted(opened),'denied':denied,
        'maximum_absolute_difference_from_original_arrays':differences,'tolerance':1e-10,
        'original_checkpoint_unchanged':True,'wall_seconds':time.perf_counter()-tick,
        'train_sha256':{str(p.relative_to(data)):hashlib.sha256(p.read_bytes()).hexdigest() for p in allowed},
        'original_config_provenance_retained':True}
    save(dest/'TRAINING_REPRODUCTION.json',record)
    if not passed:raise RuntimeError('Rebuilt arrays differ; do not replace the submitted checkpoint')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
