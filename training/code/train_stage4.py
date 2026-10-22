"""Fixed matched arithmetic feature ablation. Train only, reuse frozen OOF."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '4'
import numpy as np
from common import PROTOCOL, load_bank
from arithmetic_fusion_core import MODES, names_for, augmented_features
from fusion_core import fit_standardization
from no_test_access import install_guard
from stage3_logistic import fit_logistic
from train_stage3 import train_only_guard, SEEDS, save

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'runs/stage4')
    p.add_argument('--parser-audit-dir', type=Path, default=ROOT / 'runs/stage4_parser_frozen')
    args = p.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists(): raise FileExistsError('Never overwrite a recorded experiment')
    destination.mkdir(parents=True)
    opened = install_guard(); sys.addaudithook(train_only_guard)
    tick = time.perf_counter()
    sources = ('arithmetic_features.py', 'arithmetic_fusion_core.py', 'methods/arithmetic_fusion.py',
               'train_stage4.py', 'fusion_core.py', 'stage3_logistic.py')
    save(destination / 'budget.json', {'declared_at_utc': datetime.now(timezone.utc).isoformat(),
        'modes': MODES, 'seeds': SEEDS, 'new_base_fits': 0, 'fusion_fits': 9, 'official_validation_runs': 9,
        'meta_l2': 0.001, 'primary_seed': 20261022, 'training_split_only': True, 'test_evaluation': False,
        'plan_sha256': hashlib.sha256((ROOT.parent / 'STAGE4_PLAN.md').read_bytes()).hexdigest(),
        'source_sha256': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in sources}})
    bank = load_bank('train')
    assignments = json.loads((ROOT / 'runs/stage3/fold_assignments.json').read_text())
    assert [b['id'] for b in bank] == [a['id'] for a in assignments]
    oof_path = ROOT / 'runs/stage3/train_oof_predictions.npz'
    historical = json.loads((ROOT / 'runs/stage3/training_complete.json').read_text())
    assert hashlib.sha256(oof_path.read_bytes()).hexdigest() == historical['oof_predictions_sha256']
    with np.load(oof_path, allow_pickle=False) as m:
        oof = {k: m[k].copy() for k in m.files}
    assert oof['fold'].tolist() == [a['fold'] for a in assignments]
    parser_path = args.parser_audit_dir.resolve()
    parser_audit = json.loads((parser_path / 'summary.json').read_text())
    assert parser_audit['parser_sha256'] == hashlib.sha256((ROOT / 'arithmetic_features.py').read_bytes()).hexdigest()
    with np.load(parser_path / 'train_arithmetic_features.npz', allow_pickle=False) as m:
        arithmetic = m['features'].copy()
    assert arithmetic.shape == (len(bank), 4, 17)
    labels = oof['labels'].reshape(-1)
    records, models = [], []
    for seed in SEEDS:
        old = ROOT / f'runs/stage3/stacking_0.001_seed{seed}/checkpoint'
        with np.load(old / 'model.npz', allow_pickle=False) as m:
            arrays = {k: m[k].copy() for k in m.files if not k.startswith('meta_')}
            old_meta = {k: m[k].copy() for k in m.files if k.startswith('meta_')}
        old_config = json.loads((old / 'config.json').read_text())
        for mode in MODES:
            started = time.perf_counter()
            meta = augmented_features(bank, oof[f'mlp_seed{seed}'], oof['embedding'], mode, arithmetic)
            mean, scale = fit_standardization(meta, np.arange(len(bank)))
            x = ((meta - mean) / scale).reshape(-1, meta.shape[-1])
            theta, optimization = fit_logistic(x, labels, 0.001)
            if not optimization['converged']: raise RuntimeError('Fusion optimization failed')
            meta_arrays = dict(meta_mean=mean, meta_scale=scale, meta_weights=theta[:-1], meta_bias=np.asarray(theta[-1]))
            exact_match = None
            if mode == 'control':
                exact_match = all(np.array_equal(old_meta[k], meta_arrays[k]) for k in old_meta)
                assert exact_match, 'Re-fitted control differs from stage 3'
            folder = destination / f'{mode}_seed{seed}'
            checkpoint = folder / 'checkpoint'
            checkpoint.mkdir(parents=True)
            np.savez(checkpoint / 'model.npz', **arrays, **meta_arrays)
            config = {'method': 'soft_arithmetic_fusion_v4', 'mode': mode, 'meta_l2': 0.001, 'seed': seed,
                      'base_feature_names': old_config['base_feature_names'], 'meta_feature_names': names_for(mode),
                      'fit_split': 'train', 'base_predictions_for_meta_fit': 'frozen stage3 TRAIN question-level OOF',
                      'protocol': PROTOCOL, 'selection': 'argmax score; first on exact tie',
                      'arithmetic_is_soft_evidence_only': True}
            save(checkpoint / 'config.json', config)
            with zipfile.ZipFile(checkpoint / 'model.npz') as archive:
                size = sum(i.file_size for i in archive.infolist()) + (checkpoint / 'config.json').stat().st_size
            assert size < 16 * 2**20
            record = {'mode': mode, 'seed': seed, 'seconds': time.perf_counter() - started,
                      'features': names_for(mode), 'optimization': optimization,
                      'weights_on_standardized_features': theta[:-1].tolist(),
                      'control_parameters_exactly_match_stage3': exact_match}
            save(folder / 'training.json', record); records.append(record)
            models.append({'mode': mode, 'seed': seed, 'checkpoint': str(checkpoint.relative_to(ROOT)),
                           'inference_uncompressed_bytes': size,
                           'checkpoint_sha256': {n: hashlib.sha256((checkpoint / n).read_bytes()).hexdigest()
                                                 for n in ('model.npz', 'config.json')}})
    save(destination / 'training_complete.json', {'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'total_training_wall_seconds': time.perf_counter() - tick, 'new_base_fits': 0, 'fusion_fits': records,
        'models': models, 'no_validation_or_test_opened': True, 'opened_jsonl': sorted(opened),
        'oof_source_sha256': historical['oof_predictions_sha256'],
        'parser_sha256': parser_audit['parser_sha256'], 'parser_audit_seconds': parser_audit['seconds'],
        'reuse_note': 'Previous OOF bases already fitted; their costs are not included as new training.'})
    print(json.dumps({'new_fusion_fits': len(records), 'training_complete': True,
                      'training_seconds': time.perf_counter() - tick}),flush=True)


if __name__ == '__main__': main()
