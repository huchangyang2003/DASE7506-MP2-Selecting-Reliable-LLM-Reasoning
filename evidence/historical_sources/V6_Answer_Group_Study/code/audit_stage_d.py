"""Read-only reconciliation of four saved official outputs, resources and decision."""
from pathlib import Path
import numpy as np
import stage_d as d


def main():
    run=d.ROOT/'runs/validation/stage_d';output=run/'audit.json'
    if output.exists():raise FileExistsError(output)
    summary=d.read(run/'summary.json');freeze=d.read(run/'pre_validation_freeze.json')
    assert summary['new_official_validation_scoring_calls']==4 and summary['new_test_calls']==0
    assert {p:d.sha(p) for p in freeze['protected_before']}==freeze['protected_before']
    assert {p:d.sha(d.PROJECT/p) for p in freeze['sources_sha256']}==freeze['sources_sha256']
    bank,arrays,y=d.load_split('validation');hard=d.disagreement(bank)
    baseline=d.read(d.ROOT/'runs/baseline/stage0/validation-predictions.json')
    ids=[b['id'] for b in bank];assert baseline['ids']==ids
    base=np.asarray(baseline['selected'],dtype=int)
    base_ok=y[np.arange(len(y)),base].astype(bool)
    choices,ok={},{}
    audit=[]
    for candidate in freeze['candidates']:
        name=candidate['name'];record=summary['records'][name]
        pred=d.read(run/name/'validation-predictions.json');score=d.read(run/name/'validation-score.json')
        assert pred['ids']==ids and pred['protocol']==score['protocol']==d.PROTOCOL
        assert pred['split']==score['split']=='validation'
        selected=np.asarray(pred['selected'],dtype=int)
        assert selected.shape==(798,) and ((selected>=0)&(selected<4)).all()
        correct=y[np.arange(len(y)),selected].astype(bool)
        counts=d.metrics(correct,hard)
        assert all(counts[k]==record[k] for k in counts)
        assert score['correct']==int(correct.sum()) and score['n']==798
        assert abs(score['accuracy']-100*correct.mean())<1e-12
        for label,mask in (('all',np.ones(len(y),dtype=bool)),('disagreement',hard),('other',~hard)):
            paired=record['versus_ce'][label]
            assert paired['fixed']==int((mask&correct&~base_ok).sum())
            assert paired['broken']==int((mask&~correct&base_ok).sum())
        for tag in ('prediction_process','scoring_process'):
            process=record[tag];command=process['command']
            assert command[command.index('--split')+1]=='validation'
            assert process['returncode']==0 and not process['access']['denied']
            assert all(v['threads']<=4 for v in process['access']['native_blas_threads'])
            assert all('gsm8k_validation' in Path(p).name for p in process['access']['opened'])
        assert not any(Path(p).name=='gsm8k_validation.jsonl' and Path(p).parent.name!='candidates'
                       for p in record['prediction_process']['access']['opened'])
        cp=d.ROOT/candidate['checkpoint']
        assert {str(p.relative_to(cp)):d.sha(p) for p in cp.rglob('*')if p.is_file()}==candidate['checkpoint_sha256']
        assert d.assets(cp)==candidate['asset_bytes']==record['asset_bytes']
        choices[name],ok[name]=selected,correct
        audit.append({'name':name,'official_json_and_reconstructed_counts_match':True,
            'validation_only_fresh_commands':True,'predictor_no_reference_reads':True,
            'native_threads_at_most4':True,'checkpoint_hashes_and_assets_unchanged':True})
    c=summary['records']['C_subset'];b=summary['records']['B3_group_mean_context']
    assert c['versus_unaugmented_B3']['all']['fixed']==int((ok['C_subset']&~ok['B3_group_mean_context']).sum())
    assert c['versus_unaugmented_B3']['all']['broken']==int((~ok['C_subset']&ok['B3_group_mean_context']).sum())
    eligibility={}
    for name,r in summary['records'].items():
        checks=[r['cv_hard'][0]>1743,r['cv_hard'][1]>1749,r['disagreement']['correct']>=309,
            r['all']['correct']>=748,r['fit_and_interface_checks_pass'],
            r['asset_bytes']<=16*2**20,r['complete_validation_seconds']<=30]
        if name=='C_subset':checks += [r['both_cv_above_unaugmented_B3'],
            r['all']['correct']>=b['all']['correct'],r['disagreement']['correct']>=b['disagreement']['correct']]
        eligibility[name]=all(checks)
        assert eligibility[name]==r['passes_predeclared_gate']
    assert not any(eligibility.values()) and summary['decision']['selected_candidate']is None
    assert {p:d.sha(p) for p in freeze['protected_before']}==freeze['protected_before']
    d.save(output,{'passed':True,'candidates':audit,'eligibility_independently_reconciled':eligibility,
        'no_qualified_candidate':True,'source_freeze_and_prior_artifacts_unchanged':True,
        'new_fits':0,'new_official_runner_or_scorer_calls':0,'test_calls':0,
        'scope':'Post-hoc read-only reconciliation; did not run a prediction or official scorer again.'})
    print({'stage_d_audit':'passed','official_calls_added':0,'qualified_candidates':0})


if __name__=='__main__':main()
