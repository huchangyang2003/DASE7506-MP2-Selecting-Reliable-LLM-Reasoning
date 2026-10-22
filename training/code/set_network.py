"""Small equivariant residual networks on frozen TRAIN-OOF fusion features."""
import hashlib
import numpy as np
from objectives import loss_and_gradient

NETWORK_MODES = ('unary16', 'unary20', 'set16')


def initialize(dimension, mode, seed, dtype=np.float32):
    if mode not in NETWORK_MODES: raise ValueError('Unknown residual architecture')
    width = 20 if mode == 'unary20' else 16
    rng = np.random.default_rng(seed)
    p = {'A': (rng.standard_normal((dimension,width))/np.sqrt(dimension)).astype(dtype),
         'a': np.zeros(width,dtype=dtype),
         'B': (rng.standard_normal((width,width))/np.sqrt(width)).astype(dtype),
         'b': np.zeros(width,dtype=dtype), 'v': np.zeros(width,dtype=dtype), 'c': np.zeros((),dtype=dtype)}
    if mode == 'set16': p['C'] = np.zeros((width,width),dtype=dtype)
    return p


def forward(p, x, offset, mode, cache=False):
    z1 = x @ p['A'] + p['a']
    h = np.maximum(z1,0)
    context = (h.sum(axis=1,keepdims=True)-h)/3 if mode == 'set16' else None
    z2 = h @ p['B'] + p['b']
    if mode == 'set16': z2 = z2 + context @ p['C']
    u = np.maximum(z2,0)
    logits = offset + u @ p['v'] + p['c']
    if cache: return logits,(z1,h,context,z2,u)
    return logits


def loss_gradients(p,x,labels,offset,mode,l2=0.001):
    scores,(z1,h,context,z2,u) = forward(p,x,offset,mode,True)
    data_loss,residual = loss_and_gradient(scores,labels,'bce')
    regularized = ('A','B','v') + (('C',) if mode == 'set16' else ())
    g = {'v':np.sum(u*residual[...,None],axis=(0,1)), 'c':np.asarray(residual.sum())}
    dz2 = residual[...,None]*p['v']*(z2>0)
    width = h.shape[-1]
    g['B'] = h.reshape(-1,width).T @ dz2.reshape(-1,width)
    g['b'] = dz2.sum(axis=(0,1))
    dh = dz2 @ p['B'].T
    if mode == 'set16':
        g['C'] = context.reshape(-1,width).T @ dz2.reshape(-1,width)
        dc = dz2 @ p['C'].T
        dh = dh + (dc.sum(axis=1,keepdims=True)-dc)/3
    dz1 = dh*(z1>0)
    g['A'] = x.reshape(-1,x.shape[-1]).T @ dz1.reshape(-1,width)
    g['a'] = dz1.sum(axis=(0,1))
    for name in regularized: g[name] = g[name] + l2*p[name]
    penalty = 0.5*l2*sum(float(np.sum(p[name]**2)) for name in regularized)
    return data_loss+penalty,g,data_loss


def fit(x,labels,offset,mode,seed):
    p = initialize(x.shape[-1],mode,seed)
    common = hashlib.sha256(b''.join(p[n].tobytes() for n in ('A','a','B','b','v','c'))).hexdigest()
    initial_equal = np.array_equal(forward(p,x,offset,mode),offset)
    assert initial_equal
    sampler = np.random.default_rng(seed+1_000_000)
    first = {n:np.zeros_like(v) for n,v in p.items()}
    second = {n:np.zeros_like(v) for n,v in p.items()}
    steps = 0; schedule = hashlib.sha256(); trace=[]
    for epoch in range(60):
        order = sampler.permutation(len(labels)); schedule.update(order.tobytes())
        cumulative=0.0
        for start in range(0,len(order),128):
            ids=order[start:start+128]
            _,g,loss=loss_gradients(p,x[ids],labels[ids],offset[ids],mode)
            cumulative += loss*len(ids); steps+=1
            for n in p:
                first[n] *= 0.9; first[n] += 0.1*g[n]
                second[n] *= 0.999; second[n] += 0.001*g[n]**2
                p[n] -= 0.001*(first[n]/(1-0.9**steps))/(np.sqrt(second[n]/(1-0.999**steps))+1e-8)
        trace.append({'epoch':epoch+1,'online_mean_candidate_bce':cumulative/len(labels)})
    if not all(np.isfinite(v).all() for v in p.values()): raise ValueError('Nonfinite parameters')
    return p,{'mode':mode,'seed':seed,'epochs':60,'updates':steps,'question_batch_size':128,
              'learning_rate':0.001,'weight_l2':0.001,'trainable_parameters':sum(v.size for v in p.values()),
              'initial_output_exactly_frozen_offset':bool(initial_equal),'common_initial_sha256':common,
              'question_shuffle_sha256':schedule.hexdigest(),'trace':trace}
