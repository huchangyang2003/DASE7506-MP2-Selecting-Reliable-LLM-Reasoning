"""Fixed-CE uniform-positive CE residual; no label IO during inference."""
from pathlib import Path
import time
import numpy as np
from v7_common import PUB,read,load_asset
from linear_core import build_inputs,scores as base_scores
from pooling_core import choose
from v7_evidence import extract_evidence,SEM_NAMES,VERIFY_NAMES,REL_NAMES

FAMILIES=('R0','R1','R2')
REGS=(.03,.1)

def evidence(bank,x,family):
    if family=='R0':return x[:,:,-17:].copy()
    blocks,_=extract_evidence(bank)
    keys=('semantic','verify') if family=='R1' else ('semantic','verify','relative')
    return np.concatenate([blocks[k] for k in keys],axis=-1)

def scaler(e):
    flat=e.reshape(-1,e.shape[-1]);mean=flat.mean(0);scale=flat.std(0)
    scale[scale<1e-8]=1.
    return mean,scale

def design(e,mean,scale):return np.ascontiguousarray(np.clip((e-mean)/scale,-8,8))

def objective(u,x,offset,y,l2):
    z=offset+x@u;shift=z-z.max(1,keepdims=True);exp=np.exp(shift)
    target=y/y.sum(1,keepdims=True);den=exp.sum(1,keepdims=True)
    value=float(np.mean(np.sum(target*(np.log(den)-shift),1))+.5*l2*(u@u))
    residual=(exp/den-target)/len(y)
    grad=x.reshape(-1,x.shape[-1]).T@residual.reshape(-1)+l2*u
    return value,grad

def fit_residual(x,offset,y,l2):
    # L-BFGS/Armijo follows our frozen V3 linear_core.py; only its objective changes.
    start=time.perf_counter();u=np.zeros(x.shape[-1]);value,g=objective(u,x,offset,y,l2)
    memory=[];trace=[];evaluations=1;reason='max_iterations'
    for iteration in range(2001):
        gi=float(np.abs(g).max());trace.append({'iteration':iteration,'loss':value,'gradient_inf':gi})
        if gi<=1e-6:reason='gradient_tolerance';break
        if iteration==2000:break
        direction=g.copy();alphas=[]
        for step,change,rho in reversed(memory):
            alpha=rho*(step@direction);alphas.append(alpha);direction-=alpha*change
        if memory:
            step,change,_=memory[-1];direction*=(step@change)/(change@change)
        for (step,change,rho),alpha in zip(memory,reversed(alphas)):
            direction+=step*(alpha-rho*(change@direction))
        direction=-direction;slope=float(g@direction)
        if slope>=0:direction=-g;slope=-float(g@g);memory.clear()
        length=1.
        for _ in range(40):
            proposed=u+length*direction;nv,ng=objective(proposed,x,offset,y,l2);evaluations+=1
            if np.isfinite(nv) and np.isfinite(ng).all() and nv<=value+1e-4*length*slope:break
            length*=.5
        else:reason='line_search_failed';break
        step=proposed-u;change=ng-g;curvature=float(step@change)
        if curvature>1e-12:memory.append((step,change,1/curvature));memory=memory[-10:]
        u,value,g=proposed,nv,ng
    return u,{'converged':reason=='gradient_tolerance','stop_reason':reason,'iterations':len(trace)-1,
              'gradient_inf':float(np.abs(g).max()),'objective':value,'evaluations':evaluations,
              'trace':trace,'seconds':time.perf_counter()-start}

def candidate_scores(bank,arrays,checkpoint):
    if not bank:return np.empty((0,4),float)
    root=Path(checkpoint);c=read(root/'config.json')
    if c['method']!='v7_fixed_ce_residual' or c['alpha']!=1 or c['temperature']!=1:raise ValueError('Wrong residual configuration')
    x,names=build_inputs(bank,arrays)
    b=load_asset(root/'baseline/model.npz');e=load_asset(root/'residual.npz')
    assert names==read(root/'baseline/config.json')['feature_names']
    extra=evidence(bank,x,c['family'])
    assert extra.shape[-1]==c['extra_dimensions']
    return base_scores(x,b)+design(extra,e['mean'],e['scale'])@e['weights']

def predict(bank,arrays,checkpoint):
    if not bank:return []
    return choose(bank,candidate_scores(bank,arrays,checkpoint),1.,arrays['embeddings'])[0]

