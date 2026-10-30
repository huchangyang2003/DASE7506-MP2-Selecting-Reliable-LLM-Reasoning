"""Fresh official validation processes and descriptive paired bootstrap."""
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from experiment_io import ROOT


def run(arguments,logfile):
    start=time.perf_counter()
    result=subprocess.run([sys.executable,"-B",str(ROOT/"code/guarded_official.py")]+arguments,
                          cwd=ROOT/"code",capture_output=True,text=True,encoding="utf-8")
    seconds=time.perf_counter()-start
    logfile.parent.mkdir(parents=True,exist_ok=True)
    logfile.write_text(result.stdout+"\n"+result.stderr,encoding="utf-8")
    if result.returncode:raise RuntimeError(str(logfile)+"\n"+result.stderr)
    return seconds


def paired(new,old,hard):
    rng=np.random.default_rng(20261029);result={}
    for name,mask in (("all",np.ones(len(hard),dtype=bool)),("disagreement",hard),("other",~hard)):
        a,b=new[mask],old[mask];delta=a.astype(float)-b.astype(float)
        boots=np.concatenate([delta[rng.integers(0,len(delta),(500,len(delta)))].mean(axis=1)*100 for _ in range(10)])
        result[name]={"both_correct":int((a&b).sum()),"corrected":int((a&~b).sum()),
             "broken":int((~a&b).sum()),"both_wrong":int((~a&~b).sum()),
             "gain_percentage_points":float(delta.mean()*100),
             "descriptive_paired_bootstrap_95_interval":np.quantile(boots,[.025,.975]).tolist()}
    return result
