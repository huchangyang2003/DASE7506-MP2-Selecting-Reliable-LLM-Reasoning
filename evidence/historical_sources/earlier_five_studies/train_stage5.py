"""Fit the predeclared residual heads on TRAIN OOF features, never validation."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import numpy as np
from common import load_bank,PROTOCOL
from arithmetic_fusion_core import names_for,augmented_features
from set_network import fit,NETWORK_MODES
from no_test_access import install_guard
from train_stage3 import train_only_guard,SEEDS,save

ROOT=Path(__file__).resolve().parent
MODES=('control',)+NETWORK_MODES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'runs/stage5')
    destination=parser.parse_args().output_dir.resolve()
    if destination.exists():raise FileExistsError('Preserve previous experiments; use a new destination')
    destination.mkdir(parents=True)
    opened=install_guard();sys.addaudithook(train_only_guard)
    started=time.perf_counter()
    source_names=('train_stage5.py','set_network.py','methods/set_fusion.py','arithmetic_features.py',
                  'arithmetic_fusion_core.py','fusion_core.py','features.py','objectives.py')
    save(destination/'budget.json',{'declared_at_utc':datetime.now(timezone.utc).isoformat(),
        'modes':MODES,'seeds':SEEDS,'new_base_fits':0,'new_network_fits':9,'frozen_controls_repacked':3,
        'official_validation_runs':12,'primary_seed':20261022,'objective':'candidate BCE',
        'learning_rate':0.001,'weight_l2':0.001,'epochs':60,'question_batch_size':128,
        'frozen_offset':'stage4 arithmetic matched seed','residual_input':'same 25 OOF features, clipped standardized',
        'plan_sha256':hashlib.sha256((ROOT.parent/'STAGE5_PLAN.md').read_bytes()).hexdigest(),
        'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in source_names},
        'validation_or_test_used_for_fit':False})
    bank=load_bank('train')
    assignments=json.loads((ROOT/'runs/stage3/fold_assignments.json').read_text())
    assert [r['id'] for r in bank]==[r['id'] for r in assignments]
    source=ROOT/'runs/stage3/train_oof_predictions.npz'
    previous=json.loads((ROOT/'runs/stage3/training_complete.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==previous['oof_predictions_sha256']
    with np.load(source,allow_pickle=False) as m:oof={k:m[k].copy() for k in m.files}
    assert oof['fold'].tolist()==[r['fold'] for r in assignments]
    old_training=json.loads((ROOT/'runs/stage4_final/training_complete.json').read_text())
    assert old_training['parser_sha256']==hashlib.sha256((ROOT/'arithmetic_features.py').read_bytes()).hexdigest()
    with np.load(ROOT/'runs/stage4_parser_v2/train_arithmetic_features.npz',allow_pickle=False) as m:arithmetic=m['features'].copy()
    labels=oof['labels'];models=[];records=[]
    for seed in SEEDS:
        old=ROOT/f'runs/stage4_final/arithmetic_seed{seed}/checkpoint'
        with np.load(old/'model.npz',allow_pickle=False) as m:base={k:m[k].copy() for k in m.files}
        original_config=json.loads((old/'config.json').read_text())
        assert original_config['meta_feature_names']==names_for('arithmetic')
        meta=augmented_features(bank,oof[f'mlp_seed{seed}'],oof['embedding'],'arithmetic',arithmetic)
        standardized=(meta-base['meta_mean'])/base['meta_scale']
        offset=standardized@base['meta_weights']+base['meta_bias'].item()
        x=np.clip(standardized,-8,8).astype(np.float32)
        matching=[]
        for mode in MODES:
            folder=destination/f'{mode}_seed{seed}';checkpoint=folder/'checkpoint';checkpoint.mkdir(parents=True)
            arrays=base.copy()
            if mode!='control':
                tick=time.perf_counter();p,record=fit(x,labels,offset,mode,seed)
                record['training_seconds']=time.perf_counter()-tick
                save(folder/'training.json',record);records.append(record)
                matching.append(record)
                arrays.update({f'joint_{n}':v for n,v in p.items()})
            np.savez(checkpoint/'model.npz',**arrays)
            config={'method':'equivariant_residual_fusion_v5','mode':mode,'seed':seed,'fit_split':'train',
                    'base_feature_names':original_config['base_feature_names'],'meta_feature_names':names_for('arithmetic'),
                    'protocol':PROTOCOL,'frozen_stage4_offset':True,'question_candidates':4,
                    'selection':'argmax score; first occurrence on exact tie'}
            save(checkpoint/'config.json',config)
            with zipfile.ZipFile(checkpoint/'model.npz') as archive:
                size=sum(i.file_size for i in archive.infolist())+(checkpoint/'config.json').stat().st_size
            assert size<16*2**20
            if mode=='control':
                with np.load(checkpoint/'model.npz',allow_pickle=False) as repacked:
                    assert all(np.array_equal(repacked[k],base[k]) for k in base)
            models.append({'mode':mode,'seed':seed,'checkpoint':str(checkpoint.relative_to(ROOT)),
                           'inference_uncompressed_bytes':size,'trainable_residual_parameters':0 if mode=='control' else record['trainable_parameters'],
                           'checkpoint_sha256':{n:hashlib.sha256((checkpoint/n).read_bytes()).hexdigest() for n in ('model.npz','config.json')}})
            print(json.dumps({'finished':f'{mode}_seed{seed}','new_network_fits_so_far':len(records)}),flush=True)
        assert len({r['question_shuffle_sha256'] for r in matching})==1
        assert matching[0]['common_initial_sha256']==matching[2]['common_initial_sha256']
    save(destination/'training_complete.json',{'finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'total_training_wall_seconds':time.perf_counter()-started,'new_base_fits':0,'new_network_fits':records,
        'models':models,'opened_jsonl':sorted(opened),'no_validation_or_test_opened':True,
        'oof_source_sha256':previous['oof_predictions_sha256'],'initialization_and_batches_matched':True,
        'note':'Frozen stage4 offset and residual fit share TRAIN labels. No claim of unbiased stacked OOF accuracy.'})


if __name__=='__main__':main()
