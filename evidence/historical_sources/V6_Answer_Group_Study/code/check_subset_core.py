"""Synthetic checks for stage-C enumeration, weights and frozen-statistic isolation."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
from copy import deepcopy
import json
import numpy as np
from group_core import build_inputs, group_values, fit_statistics, objective, pair_design
from subset_core import subset_values, subset_design


def checks():
    rng = np.random.default_rng(20261030)
    vals = [['1/2', '0.5', '2', '2'], ['1', '2', '3', '4'], ['9']*4, ['1', '1', '1', '2']]
    bank = [{'id':str(i), 'question':'There are 2 and 4 units.', 'candidates':[
        {'index':j, 'text':f'c{j} #### {v}'} for j,v in enumerate(row)]} for i,row in enumerate(vals)]
    arrays = {'embeddings':rng.normal(size=(4,4,1024))}
    y = np.asarray([[1,1,0,0],[1,0,0,0],[1,1,1,1],[1,1,1,0]],dtype=float)
    x, parsed, groups = build_inputs(bank, arrays)
    gv = group_values(x, groups)
    stats = fit_statistics(x, gv, [0,1,2])
    before = {k:v.copy() for k,v in stats.items()}
    a, aw, audit = subset_design(x, groups, y, [0,1,2], 'group_mean_context', stats)
    assert len(aw) == 9+3 and audit['subset_counts_per_question'].tolist() == [6,4,0,0]
    assert np.allclose(audit['question_total_weights'], [1,1,0,0])
    assert np.allclose(aw[:9], np.repeat(1/9,9))
    assert np.all(audit['positive_mask'] & audit['negative_mask'] == 0)
    assert np.all(audit['positive_mask'] & audit['full_positive_mask'] == audit['positive_mask'])
    assert np.all(audit['negative_mask'] & audit['full_negative_mask'] == audit['negative_mask'])
    for q,pm,nm in zip(audit['owners'],audit['positive_mask'],audit['negative_mask']):
        p=[j for j in range(4) if int(pm)&(1<<j)]
        n=[j for j in range(4) if int(nm)&(1<<j)]
        assert y[q,p].all() and not y[q,n].any()
    full = next(v for v,m in subset_values(x[0], groups[0][0][1], 'group_mean_context') if int(m).bit_count()==2)
    assert np.array_equal(full, gv[0][0])
    for q,row in enumerate(groups):
        for _,members in row:
            subsets = subset_values(x[q], members, 'group_mean_context')
            assert len(subsets)==2**len(members)-1
            for v,mask in subsets:
                count=int(mask).bit_count()
                assert v[-2] == count/4
                if count==1:assert v[-1]==0
                assert np.allclose(v[1024:1030],x[q,members[0],1024:1030],atol=1e-15,rtol=1e-15)
    # All singleton groups reduce exactly to the unaugmented B design.
    b, bw, _ = pair_design(x,gv,parsed,groups,y,[1],'group_mean_context',stats)
    c, cw, _ = subset_design(x,groups,y,[1],'group_mean_context',stats)
    assert np.array_equal(b,c) and np.array_equal(bw,cw)
    changed=x.copy();changed[3]*=1000
    z,zw,_=subset_design(changed,groups,y,[0,1,2],'group_mean_context',stats)
    assert np.array_equal(a,z) and np.array_equal(aw,zw)
    assert all(np.array_equal(stats[k],v) for k,v in before.items())
    w=rng.normal(size=1032)/100
    value,gradient=objective(w,a,aw)
    errors=[]
    for j in (0,31,1024,1030,1031):
        d=np.zeros(1032);d[j]=1e-6
        numeric=(objective(w+d,a,aw)[0]-objective(w-d,a,aw)[0])/(2e-6)
        errors.append(abs(numeric-gradient[j]))
    assert max(errors)<1e-7
    # Explicit two-direction labeled loss equals the implicit positive-pair loss.
    margin=a@w
    explicit=np.dot(np.r_[aw/2,aw/2],np.logaddexp(0,-np.r_[margin,margin]))/aw.sum()+.005*(w@w)
    assert abs(value-explicit)<1e-12
    return {'passed':True,'new_optimizer_calls':0,'real_fits':0,
        'gradient_finite_difference_max_error':max(errors),'checks':[
        'all nonempty subsets including original group','canonical fractional answer groups',
        'Cartesian positive/negative subset pairs','one total weight per informative question',
        'subset masks stay inside original answer groups','labels preserved',
        'subset support/spread recomputed','numeric surface block unchanged',
        'singletons equal unaugmented design','heldout inputs cannot change fitting design',
        'passed statistics remain unchanged','weighted objective gradient','implicit reverse-pair loss identity']}


if __name__=='__main__':
    print(json.dumps(checks(),indent=2))
