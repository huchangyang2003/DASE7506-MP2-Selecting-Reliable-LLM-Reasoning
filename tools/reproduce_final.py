"""Independent test-only offline reproduction of the already-scored frozen model."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
SOURCE_DATA=ROOT/'data'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'evaluation/reproduction')
    base=parser.parse_args().output_dir.resolve()
    frozen=json.loads((ROOT/'FREEZE.json').read_text())
    for name,r in frozen['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==r['sha256'],name
    if base.exists():raise FileExistsError('Preserve original reproduction evidence; choose a new --output-dir')
    isolated=base/'isolated'
    isolated.mkdir(parents=True)
    copied=[]
    for name in frozen['files']:
        target=isolated/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target);copied.append(name)
    for name in ('manifest.json','LICENSE.txt','NOTICE.txt','gsm8k_test.jsonl','candidates/gsm8k_test.jsonl','candidates/gsm8k_test.npz'):
        target=isolated/'code/data'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(SOURCE_DATA/name,target);copied.append('code/data/'+name)
    # Python audit guard disallows networking and user-file reads outside this isolated tree.
    # Interpreter/NumPy library reads remain allowed. Only inference is denied access to gold.
    (isolated/'code/offline_runner.py').write_text('''import os,runpy,socket,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
libs=Path(sys.base_prefix).resolve()
venv=Path(sys.prefix).resolve()
def blocked(*a,**k):raise RuntimeError("Offline reproduction: network blocked")
socket.create_connection=blocked;socket.getaddrinfo=blocked
socket.socket.connect=blocked;socket.socket.connect_ex=blocked
prediction=sys.argv[1]=='predict.py'
def audit(event,args):
 if event.startswith('socket.'):blocked()
 if event=='open' and isinstance(args[0],str):
  p=Path(args[0]).resolve()
  if not any(p.is_relative_to(base) for base in (root,libs,venv)):
   raise PermissionError('Read outside isolated tree/runtime: '+str(p))
  if prediction and p.name=='gsm8k_test.jsonl' and p.parent.name=='data':
   raise PermissionError('Prediction cannot read gold reference')
sys.addaudithook(audit)
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
''',encoding='utf-8')
    env={**os.environ,'PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1',
         'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONIOENCODING':'utf-8'}
    commands=[['predict.py','--method','submission.py','--checkpoint','../checkpoint','--split','test','--output','../test-predictions.json'],
              ['score_predictions.py','--split','test','--predictions','../test-predictions.json','--output','../test-score.json']]
    elapsed=[]
    for i,command in enumerate(commands):
        tick=time.perf_counter()
        result=subprocess.run([sys.executable,'-B','offline_runner.py',*command],cwd=isolated/'code',env=env,
                              capture_output=True,text=True,encoding='utf-8',timeout=30)
        elapsed.append(time.perf_counter()-tick)
        (isolated.parent/f'process-{i}.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError(result.stderr)
    original=json.loads((ROOT/'evaluation/final_test/test-predictions.json').read_text())
    repeated=json.loads((isolated/'test-predictions.json').read_text())
    for key in ('protocol','split','ids','selected'):assert original[key]==repeated[key],key
    score=json.loads((isolated/'test-score.json').read_text())
    expected=json.loads((ROOT/'evaluation/final_test/test-score.json').read_text())
    for key in ('protocol','split','correct','n','accuracy'):assert score[key]==expected[key],key
    assert not list((isolated/'code/data').rglob('*train*')) and not list((isolated/'code/data').rglob('*validation*'))
    record={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'passed':True,
        'all_467_ids_and_choices_exactly_match':True,'official_score':score,
        'offline_guard':True,'user_file_reads_restricted_to_isolated_tree':True,
        'gold_reference_read_denied_to_prediction_process':True,'contains_no_training_or_validation_files':True,
        'prediction_process_wall_seconds':elapsed[0],'scoring_process_wall_seconds':elapsed[1],
        'total_process_wall_seconds':sum(elapsed),'copied_files':copied,
        'reference_cpu_eligibility_confirmed':False,'purpose':'Reproduction only, no method selection or training'}
    (isolated.parent/'VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k!='copied_files'},indent=2))


if __name__=='__main__':main()
