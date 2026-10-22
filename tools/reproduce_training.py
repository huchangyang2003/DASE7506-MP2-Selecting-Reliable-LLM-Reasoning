"""Optional from-scratch TRAIN-only reproduction in a new tree; never tune on test."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'training'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,required=True)
    args=p.parse_args();dest=args.destination.resolve()
    if dest.exists():raise FileExistsError('Use a new directory; preserve all recorded work')
    code=dest/'code';code.mkdir(parents=True)
    for item in (SOURCE/'code').glob('*.py'):shutil.copyfile(item,code/item.name)
    shutil.copytree(SOURCE/'code/methods',code/'methods',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('STAGE3_PLAN.md','STAGE4_PLAN.md'):shutil.copyfile(SOURCE/name,dest/name)
    train_data=ROOT/'data'
    for name in ('manifest.json','LICENSE.txt','NOTICE.txt','gsm8k_train.jsonl','candidates/gsm8k_train.jsonl','candidates/gsm8k_train.npz'):
        target=code/'data'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(train_data/name,target)
    commands=[['train_ranker.py','--objective','bce','--seed',str(seed),'--output-dir',f'runs/stage1/bce_seed{seed}']
              for seed in (20261022,20261023,20261024)]
    commands += [['train_stage3.py'],['audit_arithmetic_train.py','--output-dir','runs/stage4_parser_v2'],
                 ['train_stage4.py','--output-dir','runs/stage4_final','--parser-audit-dir','runs/stage4_parser_v2']]
    records=[]
    for i,command in enumerate(commands):
        tick=time.perf_counter();result=subprocess.run([sys.executable,'-B',*command],cwd=code,
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8'},capture_output=True,text=True,encoding='utf-8',timeout=300)
        (dest/f'process-{i}.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        records.append({'command':command,'seconds':time.perf_counter()-tick,'return_code':result.returncode})
        if result.returncode:raise RuntimeError(result.stderr)
    import numpy as np
    rebuilt=code/'runs/stage4_final/arithmetic_seed20261022/checkpoint'
    with np.load(rebuilt/'model.npz',allow_pickle=False) as a,np.load(ROOT/'checkpoint/model.npz',allow_pickle=False) as b:
        exact=all(np.array_equal(a[k],b[k]) for k in b.files)
        max_abs=max(float(np.max(np.abs(a[k]-b[k]))) for k in b.files)
    record={'commands':records,'training_only':True,'new_base_and_fusion_fits':42,
        'selected_checkpoint':str(rebuilt),'frozen_arrays_exactly_match':exact,'max_parameter_abs_difference':max_abs,
        'note':'Exact floating-point matching depends on hardware/NumPy/BLAS. No parameters are selected or changed by this comparison.',
        'test_or_validation_opened':False}
    (dest/'VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
