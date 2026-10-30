"""Run one unchanged official validation process with bank and thread guards."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
import ctypes
from pathlib import Path
import runpy
import sys
import numpy as np

OLD_CODE=Path(__file__).resolve().parents[2]/'V3_Block_Regularization/code'
sys.path.insert(0,str(OLD_CODE))
from experiment_io import guard,save


def main():
    runner,args=sys.argv[1],sys.argv[2:]
    if runner not in ('predict.py','score_predictions.py'):
        raise ValueError('Only the two official validation runners are permitted')
    if '--split' not in args or args[args.index('--split')+1]!='validation':
        raise ValueError('Explicit validation split required')
    data=Path(args[args.index('--data-dir')+1]).resolve()
    role='validation_predict' if runner=='predict.py' else 'validation_score'
    opened,denied=guard(role)
    allowed={str((data/'candidates/gsm8k_validation.jsonl').resolve()).lower(),
             str((data/'candidates/gsm8k_validation.npz').resolve()).lower()}
    if role=='validation_score':allowed.add(str((data/'gsm8k_validation.jsonl').resolve()).lower())
    def only_validation(event,values):
        if event=='open' and isinstance(values[0],(str,bytes)):
            path=Path(os.fsdecode(values[0])).resolve()
            if path.name.lower().startswith('gsm8k_') and str(path).lower() not in allowed:
                raise PermissionError('Unexpected bank/reference in validation process')
    sys.addaudithook(only_validation)
    native=[]
    for path in (Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*.dll'):
        lib=ctypes.CDLL(str(path));function=getattr(lib,'scipy_openblas_get_num_threads64_')
        function.restype=ctypes.c_int
        native.append({'library':path.name,'threads':function()})
    assert native and all(row['threads']<=4 for row in native)
    sys.argv=[runner]+args
    runpy.run_path(str(OLD_CODE/runner),run_name='__main__')
    output=Path(args[args.index('--output')+1])
    save(output.with_suffix('.access.json'),{'role':role,'opened':sorted(opened),'denied':denied,
        'native_blas_threads':native,'thread_environment':{k:os.environ[k] for k in
            ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        'official_source':str(OLD_CODE/runner)})


if __name__=='__main__':main()
