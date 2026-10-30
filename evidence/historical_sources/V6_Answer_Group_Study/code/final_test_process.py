"""One authorized official TEST process; prediction cannot open gold."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
import ctypes
import json
from pathlib import Path
import runpy
import sys
import numpy as np

OFFICIAL=Path(__file__).resolve().parents[2]/'DASE7506-MP2-main/code'
sys.path.insert(0,str(OFFICIAL))


def main():
    runner,args=sys.argv[1],sys.argv[2:]
    assert runner in ('predict.py','score_predictions.py')
    assert args[args.index('--split')+1]=='test'
    data=Path(args[args.index('--data-dir')+1]).resolve()
    role='test_predict' if runner=='predict.py' else 'test_score'
    allowed={(data/'candidates/gsm8k_test.jsonl').resolve(),
             (data/'candidates/gsm8k_test.npz').resolve()}
    if role=='test_score':allowed.add((data/'gsm8k_test.jsonl').resolve())
    opened=set();denied=[]
    def audit(event,values):
        if event.startswith('socket.connect'):raise PermissionError('Offline evaluation')
        if event=='open' and isinstance(values[0],(str,bytes)):
            p=Path(os.fsdecode(values[0])).resolve()
            if p.name.lower().startswith('gsm8k_'):
                if p not in allowed:
                    denied.append(str(p));raise PermissionError('Unexpected bank/reference: '+str(p))
                opened.add(str(p))
    sys.addaudithook(audit)
    native=[]
    for p in (Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*.dll'):
        lib=ctypes.CDLL(str(p));function=getattr(lib,'scipy_openblas_get_num_threads64_')
        function.restype=ctypes.c_int
        native.append({'library':p.name,'threads':function()})
    assert native and all(r['threads']<=4 for r in native)
    sys.argv=[runner]+args
    runpy.run_path(str(OFFICIAL/runner),run_name='__main__')
    output=Path(args[args.index('--output')+1])
    output.with_suffix('.access.json').write_text(json.dumps({
        'role':role,'opened':sorted(opened),'denied':denied,
        'native_blas_threads':native,'official_source':str(OFFICIAL/runner),
        'thread_environment':{k:os.environ[k] for k in
            ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}},indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
