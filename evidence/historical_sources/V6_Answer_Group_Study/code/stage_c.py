"""One frozen stage-C augmentation, 11 real fits, TRAIN only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import time
import numpy as np
import stage_b as sb
from check_subset_core import checks
from subset_core import subset_design
from group_core import build_inputs,group_values,fit,objective,load_model,model_scores,choose_scores
from experiment_io import load_split,disagreement,metrics

ROOT,PROJECT=sb.ROOT,sb.PROJECT
B_RUN=ROOT/'runs/group_models/stage_b'
MODE='group_mean_context'
FITS=[]


def reused_stats(partition,fold,fit_rows,held,out):
    source=B_RUN/'full_statistics.npz' if fold is None else B_RUN/partition/f'fold_{fold}_statistics.npz'
    target=out/'statistics_from_B.npz'
    shutil.copyfile(source,target)
    assert sb.io.sha(source)==sb.io.sha(target)
    with np.load(source,allow_pickle=False) as z:
        stats={k:z[k].copy() for k in z.files}
    assert np.array_equal(stats['train_rows'],fit_rows)
    if held is not None:assert np.array_equal(stats['held_rows'],held)
    baseline_cp=B_RUN/'full'/MODE/'checkpoint' if fold is None else B_RUN/partition/MODE/f'fold_{fold}/checkpoint'
    baseline=load_model(baseline_cp)
    assert np.array_equal(stats['mean'],baseline['mean'])
    assert np.array_equal(stats['scale'],baseline['scale'])
    sb.io.save(out/'statistics_provenance.json',{'source':str(source.relative_to(ROOT)),
        'source_sha256':sb.io.sha(source),'copy_sha256':sb.io.sha(target),
        'reestimated':False,'baseline_checkpoint':str(baseline_cp.relative_to(ROOT))})
    return stats


def fit_one(bank,x,parsed,groups,y,fit_rows,stats,out,run,tag):
    tick=time.perf_counter()
    X,weights,audit=subset_design(x,groups,y,fit_rows,MODE,stats)
    design_seconds=time.perf_counter()-tick
    np.savez_compressed(out/'pair_audit.npz',**audit)
    record={'call':len(FITS)+1,'tag':tag,'state':'started','positive_pair_rows':len(weights),
        'equivalent_ordered_pair_rows':2*len(weights),
        'informative_questions':len(audit['informative_questions']),
        'original_group_pair_rows':int(audit['original_group_pair_counts'].sum()),
        'generated_subset_groups':int(audit['subset_counts_per_question'].sum()),
        'design_seconds':design_seconds,'statistics_sha256':sb.io.sha(out/'statistics_from_B.npz')}
    FITS.append(record);sb.io.save(run/'fit_progress.json',FITS)
    tick=time.perf_counter();w,opt=fit(X,weights,maxiter=2000,gtol=1e-6)
    seconds=time.perf_counter()-tick
    value,gradient=objective(w,X,weights)
    record.update({'state':'completed','converged':opt['converged'],'fit_seconds':seconds,
        'iterations':opt['iterations'],'gradient_inf':float(np.abs(gradient).max())})
    sb.io.save(out/'optimization.json',opt);sb.io.save(run/'fit_progress.json',FITS)
    if not opt['converged'] or np.abs(gradient).max()>1e-6:
        raise RuntimeError('Nonconverged augmentation fit: '+tag)
    assert abs(value-opt['final_loss'])<1e-12
    model={'mode':MODE,'weights':w,'mean':stats['mean'].copy(),'scale':stats['scale'].copy()}
    cp=out/'checkpoint';sb.export_model(cp,model)
    config=sb.io.read(cp/'config.json')
    config.update({'training_augmentation':'all_nonempty_subsets_within_same_numeric_answer',
        'scaler_reestimated_on_augmentation':False,'statistics_source_sha256':record['statistics_sha256'],
        'inference_uses_full_original_groups':True,'same_original_question_total_weight':True})
    sb.io.save(cp/'config.json',config)
    reload=load_model(cp)
    assert all(np.array_equal(reload[k],model[k]) for k in ('weights','mean','scale'))
    record['checkpoint']=str(cp.relative_to(ROOT))
    record['checkpoint_sha256']={p.name:sb.io.sha(p) for p in cp.iterdir()}
    sb.io.save(run/'fit_progress.json',FITS)
    print(f"{tag}: pairs={len(weights)} iter={opt['iterations']} grad={record['gradient_inf']:.3g} seconds={seconds:.3f}",flush=True)
    return model


def cv(bank,arrays,x,parsed,groups,gv,y,hard,partition,run):
    out=run/partition;out.mkdir()
    with np.load(B_RUN/partition/'oof_scores.npz',allow_pickle=False) as z:
        folds=z['folds'].copy();baseline_choices=z['selected_'+MODE].copy()
        assert np.array_equal(y,z['labels']) and np.array_equal(hard,z['hard'])
    with np.load(sb.OLD_RUN/partition/'oof_scores.npz',allow_pickle=False) as z:
        assert np.array_equal(folds,z['fold'])
        ce=z[sb.io.NAME].argmax(1)
    from fusion_core import grouped_folds
    seed=20261022 if partition=='primary' else 20261029
    assert np.array_equal(folds,grouped_folds(bank,seed=seed))
    values=np.full((len(bank),4),np.nan);selected=np.full(len(bank),-1,dtype=int)
    coverage=np.zeros(len(bank),dtype=int)
    for fold in range(5):
        fit_rows,held=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
        assert not ({' '.join(bank[i]['question'].split()) for i in fit_rows}&
                    {' '.join(bank[i]['question'].split()) for i in held})
        folder=out/f'fold_{fold}';folder.mkdir()
        stats=reused_stats(partition,fold,fit_rows,held,folder)
        model=fit_one(bank,x,parsed,groups,y,fit_rows,stats,folder,run,f'{partition}/fold{fold}')
        hb=[bank[i] for i in held];hp=[parsed[i] for i in held]
        hg=[groups[i] for i in held];hv=[gv[i] for i in held]
        scores=model_scores(hb,x[held],hp,hg,hv,model)
        choices=choose_scores(hb,x[held],hp,scores)
        values[held]=scores;selected[held]=choices;coverage[held]+=1
    assert np.isfinite(values).all() and np.all(coverage==1)
    assert ((selected>=0)&(selected<4)).all()
    correct=y[np.arange(len(y)),selected].astype(bool)
    patterns={}
    for i,row in enumerate(parsed):
        counts={}
        for v in row:
            if v is not None:counts[v]=counts.get(v,0)+1
        key=','.join(map(str,sorted(counts.values(),reverse=True))) or 'none'
        item=patterns.setdefault(key,{'n':0,'correct':0});item['n']+=1;item['correct']+=int(correct[i])
    np.savez_compressed(out/'oof_scores.npz',scores=values,selected=selected,folds=folds,labels=y,hard=hard)
    result={'seed':seed,**metrics(correct,hard),
        'versus_ce':sb.io.comparisons(selected,ce,parsed,y,hard),
        'versus_unaugmented_B3':sb.io.comparisons(selected,baseline_choices,parsed,y,hard),
        'by_fold':[{'fold':f,**metrics(correct[folds==f],hard[folds==f])} for f in range(5)],
        'by_valid_answer_support_pattern':patterns}
    sb.io.save(out/'summary.json',result)
    return result


def execute(run):
    tick=time.perf_counter()
    status=sb.io.read(ROOT/'STATUS.json')
    assert status['stage_b']=='completed' and status['stage_c']=='not_started'
    assert status['stage_b_selected_group_mode']==MODE and status['normal_fit_budget_remaining']==11
    assert sb.io.read(B_RUN/'selection_before_confirmation.json')['selected_group_mode']==MODE
    assert sb.io.read(B_RUN/'audit.json')['passed']
    protected=sb.io.protected_snapshot()
    source={name:sb.io.sha(ROOT/'code'/name) for name in
        ('stage_c.py','subset_core.py','check_subset_core.py','group_core.py','stage_b.py','stage_a.py')}
    sb.io.save(run/'started.json',{'started_utc':datetime.now(timezone.utc).isoformat(),
        'plan_sha256':sb.io.sha(ROOT/'PLAN.md'),'budget_sha256':sb.io.sha(ROOT/'BUDGET.json'),
        'sources_sha256':source,'protected_before':protected,'mode':MODE,'max_real_fits':11,
        'fixed_B_selection_sha256':sb.io.sha(B_RUN/'selection_before_confirmation.json'),
        'native_blas_threads':sb.blas_threads(),'new_validation_calls':0,'new_test_calls':0,
        'statistics_policy':'Byte-copy each original B3 statistic file; no fitting on augmented subsets.',
        'subset_weight':'1/(original positive-negative group pairs * positive subset count * negative subset count)'})
    synthetic=checks();sb.io.save(run/'synthetic_checks.json',synthetic)
    bank,arrays,y=load_split('train');hard=disagreement(bank)
    x,parsed,groups=build_inputs(bank,arrays);gv=group_values(x,groups)
    assert len(bank)==4755 and int(hard.sum())==2114
    sb.io.save(run/'inputs.json',{'questions':len(bank),'hard':int(hard.sum()),
        'train_data_sha256':{p:sb.io.sha(sb.io.DATA/p) for p in
            ('gsm8k_train.jsonl','candidates/gsm8k_train.jsonl','candidates/gsm8k_train.npz')},
        'dimension':1032,'new_candidates_generated':False,'baseline_retrained':False})
    primary=cv(bank,arrays,x,parsed,groups,gv,y,hard,'primary',run)
    confirmation=cv(bank,arrays,x,parsed,groups,gv,y,hard,'confirmation',run)
    folder=run/'full';folder.mkdir()
    fit_rows=np.arange(len(bank));stats=reused_stats('full',None,fit_rows,None,folder)
    fit_one(bank,x,parsed,groups,y,fit_rows,stats,folder,run,'full_train')
    assert len(FITS)==11 and all(f['converged'] for f in FITS)
    assert {p:sb.io.sha(p) for p in protected}==protected
    assert {name:sb.io.sha(ROOT/'code'/name) for name in source}==source
    assert not sb.io.DENIED
    result={'stage':'C','status':'completed','finished_utc':datetime.now(timezone.utc).isoformat(),
        'mode':MODE,'augmentation':'all_nonempty_subsets_within_same_numeric_answer',
        'primary':primary,'confirmation':confirmation,
        'both_cv_above_ce':primary['disagreement']['correct']>1743 and confirmation['disagreement']['correct']>1749,
        'both_cv_above_unaugmented_B3':all(r['versus_unaugmented_B3']['disagreement']['net_correct']>0
                                         for r in (primary,confirmation)),
        'checkpoint':str((folder/'checkpoint').relative_to(ROOT)),
        'checkpoint_sha256':FITS[-1]['checkpoint_sha256'],
        'new_real_fits':11,'all11_converged':True,'baseline_retrained':False,
        'max_gradient_inf':max(f['gradient_inf'] for f in FITS),
        'fit_seconds_sum':sum(f['fit_seconds'] for f in FITS),
        'design_seconds_sum':sum(f['design_seconds'] for f in FITS),
        'wall_seconds':time.perf_counter()-tick,'official_validation_calls':0,'official_test_calls':0,
        'new_synthetic_optimizer_calls':0,'protected_artifacts_unchanged':len(protected),
        'native_blas_threads':sb.blas_threads(),'cpu':sb.cpu_name(),
        'bank_accesses':sorted(p for p in sb.io.OPENED if Path(p).name.startswith('gsm8k_')),
        'denied_accesses':sb.io.DENIED,'validation_gate':'not_evaluated_until_stage_D',
        'note':'The one B3 configuration was fixed before this experiment; no reselection or unbiased CV claim.'}
    sb.io.save(run/'summary.json',result)
    print({'stage_c':'completed','primary_hard':primary['disagreement']['correct'],
        'confirmation_hard':confirmation['disagreement']['correct'],'real_fits':11,
        'both_cv_above_ce':result['both_cv_above_ce'],
        'both_cv_above_B3':result['both_cv_above_unaugmented_B3']},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/subset_augmentation/stage_c')
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT) or out.exists():raise ValueError('Use a fresh V6 output directory')
    out.mkdir(parents=True)
    try:execute(out)
    except Exception as error:
        sb.io.save(out/'failure.json',{'type':type(error).__name__,'message':str(error),
            'real_fit_calls':len(FITS),'records':FITS,'denied_accesses':sb.io.DENIED})
        raise


if __name__=='__main__':main()
