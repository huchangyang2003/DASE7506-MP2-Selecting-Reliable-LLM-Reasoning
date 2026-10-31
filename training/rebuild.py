"""Optional fixed TRAIN-only R1 reproduction: one CE fit and one residual fit, no search."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
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
from linear_core import build_inputs,fit_scaler,fit,scores
from residual_core import evidence,scaler,design,fit_residual

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    args=parser.parse_args();dest=args.destination.resolve();data=args.data_dir.resolve()
    if dest.exists():raise FileExistsError('Use a fresh destination; never replace original evidence/checkpoint')
    dest.mkdir(parents=True);start=time.perf_counter();opened=set();denied=[]
    allowed={(data/'gsm8k_train.jsonl').resolve(),(data/'candidates/gsm8k_train.jsonl').resolve(),(data/'candidates/gsm8k_train.npz').resolve()}
    def audit(event,values):
        if event.startswith('socket.connect'):raise PermissionError('Offline fixed training reproduction')
        if event=='open' and isinstance(values[0],(str,bytes)):
            path=Path(os.fsdecode(values[0])).resolve()
            if path.name.lower().startswith('gsm8k_'):
                if path not in allowed:denied.append(str(path));raise PermissionError('Only TRAIN is permitted')
                opened.add(str(path))
    sys.addaudithook(audit)
    bank,arrays=load_inputs('train',data);refs=read_jsonl(data/'gsm8k_train.jsonl')
    assert len(bank)==len(refs)==4755
    assert all(b['id']==r['id'] and b['question']==r['question'] for b,r in zip(bank,refs))
    y=np.asarray([[int((a:=parse_answer(c['text'])) is not None and a==parse_answer(r['answer']))
                   for c in b['candidates']] for b,r in zip(bank,refs)],float)
    assert np.all(y.sum(1)>0)
    x,names=build_inputs(bank,arrays);assert x.shape==(4755,4,1071)
    assert names==read(ROOT/'checkpoint/baseline/config.json')['feature_names']
    mean,scale=fit_scaler(x)
    save(dest/'CALL_LEDGER.json',{'fit_calls':1,'current':'fixed full-TRAIN CE','validation_calls':0,'test_calls':0})
    theta,ce_record=fit(np.ascontiguousarray((x-mean)/scale),y,'uniform_ce',.01,.01)
    save(dest/'CE_OPTIMIZATION.json',ce_record)
    assert ce_record['converged'] and ce_record['gradient_inf']<=1e-6
    baseline={'mean':mean,'scale':scale,'weights':theta[:-1],'bias':np.asarray(theta[-1])}
    extra=evidence(bank,x,'R1');assert extra.shape==(4755,4,50)
    emean,escale=scaler(extra)
    save(dest/'CALL_LEDGER.json',{'fit_calls':2,'current':'fixed full-TRAIN R1 residual','validation_calls':0,'test_calls':0})
    weights,residual_record=fit_residual(design(extra,emean,escale),scores(x,baseline),y,.03)
    save(dest/'RESIDUAL_OPTIMIZATION.json',residual_record)
    assert residual_record['converged'] and residual_record['gradient_inf']<=1e-6
    checkpoint=dest/'checkpoint';(checkpoint/'baseline').mkdir(parents=True)
    np.savez(checkpoint/'baseline/model.npz',**baseline)
    np.savez(checkpoint/'residual.npz',mean=emean,scale=escale,weights=weights)
    for rel in ['config.json','baseline/config.json']:shutil.copyfile(ROOT/'checkpoint'/rel,checkpoint/rel)
    differences={}
    for rel in ['baseline/model.npz','residual.npz']:
        with np.load(ROOT/'checkpoint'/rel,allow_pickle=False) as original,np.load(checkpoint/rel,allow_pickle=False) as rebuilt:
            assert set(original.files)==set(rebuilt.files)
            differences[rel]={k:float(np.max(np.abs(original[k]-rebuilt[k]))) for k in original.files}
    passed=all(v<=1e-10 for block in differences.values() for v in block.values())
    result={'passed':passed,'purpose':'Fixed selected-recipe reproduction, not model selection',
      'fit_calls':2,'validation_calls':0,'test_calls':0,'whole_train_questions':len(bank),
      'maximum_absolute_array_differences':differences,'tolerance':1e-10,
      'original_checkpoint_unchanged':True,'opened':sorted(opened),'denied':denied,
      'train_sha256':{str(p.relative_to(data)):hashlib.sha256(p.read_bytes()).hexdigest() for p in allowed},
      'wall_seconds':time.perf_counter()-start}
    save(dest/'TRAINING_REPRODUCTION.json',result)
    if not passed:raise RuntimeError('Numerical mismatch; inspect platform/BLAS without replacing or retuning the shipped checkpoint')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
