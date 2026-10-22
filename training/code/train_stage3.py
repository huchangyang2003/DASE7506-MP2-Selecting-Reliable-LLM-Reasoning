"""Fit all bases and fusion weights on TRAIN only, before validation is opened."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import numpy as np
from common import DATA, PROTOCOL, load_inputs, parse_answer, read_jsonl
from features import build_features
from fusion_core import FEATURE_NAMES, grouped_folds, fit_standardization, meta_features, mlp_logits
from no_test_access import install_guard
from stage3_logistic import fit_logistic
from train_ranker import fit

ROOT = Path(__file__).resolve().parent
SEEDS = (20261022, 20261023, 20261024)
REGULARIZATIONS = (0.001, 0.01, 0.1)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def train_only_guard(event, arguments):
    if event == 'open' and isinstance(arguments[0], str):
        if Path(arguments[0]).name.lower().startswith('gsm8k_validation'):
            raise PermissionError('Fusion training cannot open validation inputs or labels')


def linear_fit(features, labels, train_ids):
    mean, scale = fit_standardization(features, train_ids)
    x = ((features[train_ids] - mean) / scale).reshape(-1, features.shape[-1])
    theta, record = fit_logistic(x, labels[train_ids].reshape(-1), 0.01)
    if not record['converged']:
        raise RuntimeError(f'Linear base did not converge: {record["stop_reason"]}')
    return {'emb_mean': mean, 'emb_scale': scale, 'emb_weights': theta[:-1], 'emb_bias': theta[-1]}, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/stage3')
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError('Use a new destination; never overwrite an experiment')
    destination.mkdir(parents=True)
    opened = install_guard()
    sys.addaudithook(train_only_guard)
    started = time.perf_counter()
    source_names = ('train_stage3.py', 'fusion_core.py', 'methods/fusion_model.py', 'stage3_logistic.py',
                    'train_ranker.py', 'objectives.py', 'features.py')
    save(destination / 'budget.json', {
        'declared_at_utc': datetime.now(timezone.utc).isoformat(), 'folds': 5, 'fold_seed': 20261022,
        'seeds': SEEDS, 'meta_l2': REGULARIZATIONS, 'new_base_fits': 21, 'meta_fits': 9,
        'official_validation_runs': 15, 'input_only_meta_features': FEATURE_NAMES,
        'base_embedding_l2': 0.01, 'train_split_only': True, 'test_evaluation': False,
        'plan_sha256': hashlib.sha256((ROOT.parent / 'STAGE3_PLAN.md').read_bytes()).hexdigest(),
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names}})
    bank, arrays = load_inputs('train')
    refs = read_jsonl(DATA / 'gsm8k_train.jsonl')
    if len(bank) != len(refs) or any(b['id'] != ref['id'] or b['question'] != ref['question'] for b, ref in zip(bank, refs)):
        raise ValueError('TRAIN alignment differs')
    labels = np.array([[int((a := parse_answer(c['text'])) is not None and a == parse_answer(ref['answer']))
                        for c in b['candidates']] for b, ref in zip(bank, refs)], dtype=np.float32)
    assert labels.any(axis=1).all()
    features, names = build_features(bank, arrays, 'group')
    embeddings = np.asarray(arrays['embeddings'], dtype=float)
    folds = grouped_folds(bank)
    save(destination / 'fold_assignments.json', [{'id': b['id'], 'fold': int(f)} for b, f in zip(bank, folds)])
    oof_linear = np.full(labels.shape, np.nan, dtype=float)
    oof_mlp = {seed: np.full(labels.shape, np.nan, dtype=np.float32) for seed in SEEDS}
    base_records, fold_records = [], []
    for fold in range(5):
        train_ids, held_ids = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit_questions = {' '.join(bank[i]['question'].split()) for i in train_ids}
        held_questions = {' '.join(bank[i]['question'].split()) for i in held_ids}
        assert not fit_questions & held_questions
        folder = destination / f'fold_{fold}'
        folder.mkdir()
        tick = time.perf_counter()
        linear, record = linear_fit(embeddings, labels, train_ids)
        oof_linear[held_ids] = ((embeddings[held_ids] - linear['emb_mean']) / linear['emb_scale']) @ linear['emb_weights'] + linear['emb_bias']
        np.savez(folder / 'embedding.npz', **linear)
        save(folder / 'embedding-training.json', record)
        base_records.append({'model': 'embedding', 'fold': fold, 'seed': None, 'seconds': time.perf_counter() - tick,
                             'converged': record['converged']})
        mean, scale = fit_standardization(features, train_ids)
        train_features = np.clip((features[train_ids] - mean) / scale, -8, 8).astype(np.float32)
        fold_records.append({'fold': fold, 'fit_questions': len(train_ids), 'heldout_questions': len(held_ids),
                             'normalized_question_overlap': 0, 'fit_row_indices': train_ids.tolist(),
                             'heldout_row_indices': held_ids.tolist()})
        for seed in SEEDS:
            tick = time.perf_counter()
            params, record = fit(train_features, labels[train_ids], 'bce', seed)
            oof_mlp[seed][held_ids] = mlp_logits(features[held_ids], mean, scale, params)
            np.savez(folder / f'mlp_seed{seed}.npz', mean=mean, scale=scale, **params)
            save(folder / f'mlp_seed{seed}-training.json', record)
            base_records.append({'model': 'mlp', 'fold': fold, 'seed': seed, 'seconds': time.perf_counter() - tick,
                                 'epochs': record['epochs'], 'updates': record['updates']})
        save(destination / 'base_fit_progress.json', base_records)
        print(json.dumps({'completed_fold': fold, 'new_base_fits_done': len(base_records)}), flush=True)
    assert np.isfinite(oof_linear).all() and all(np.isfinite(a).all() for a in oof_mlp.values())
    save(destination / 'fold_audit.json', fold_records)
    np.savez(destination / 'train_oof_predictions.npz', fold=folds, labels=labels, embedding=oof_linear,
             **{f'mlp_seed{seed}': a for seed, a in oof_mlp.items()})
    tick = time.perf_counter()
    full_linear, linear_record = linear_fit(embeddings, labels, np.arange(len(bank)))
    base_records.append({'model': 'embedding', 'fold': 'full_train', 'seed': None,
                         'seconds': time.perf_counter() - tick, 'converged': linear_record['converged']})
    save(destination / 'full-embedding-training.json', linear_record)
    all_models, meta_records = [], []
    for seed in SEEDS:
        historical = ROOT / f'runs/stage1/bce_seed{seed}/checkpoint'
        with np.load(historical / 'model.npz', allow_pickle=False) as model:
            full_mlp = {key: model[key].copy() for key in model.files}
        meta = meta_features(bank, oof_mlp[seed], oof_linear)
        meta_mean, meta_scale = fit_standardization(meta, np.arange(len(bank)))
        x = ((meta - meta_mean) / meta_scale).reshape(-1, len(FEATURE_NAMES))
        for mode in ('mlp_control', 'equal_average', 'stacking_0.001', 'stacking_0.01', 'stacking_0.1'):
            folder = destination / f'{mode}_seed{seed}'
            checkpoint = folder / 'checkpoint'
            checkpoint.mkdir(parents=True)
            model_arrays = {**full_mlp, **full_linear}
            l2 = None
            if mode.startswith('stacking_'):
                l2 = float(mode.split('_')[1])
                tick = time.perf_counter()
                theta, optimization = fit_logistic(x, labels.reshape(-1), l2)
                if not optimization['converged']:
                    raise RuntimeError('Meta optimizer did not converge')
                record = {'mode': mode, 'seed': seed, 'seconds': time.perf_counter() - tick,
                          'optimization': optimization, 'weights_on_standardized_features': theta[:-1].tolist(),
                          'intercept': float(theta[-1]), 'fit_rows': len(x), 'fit_source': 'TRAIN OOF only'}
                meta_records.append(record)
                save(folder / 'meta-training.json', record)
                model_arrays.update(meta_mean=meta_mean, meta_scale=meta_scale,
                                    meta_weights=theta[:-1], meta_bias=theta[-1])
            np.savez(checkpoint / 'model.npz', **model_arrays)
            config = {'method': 'train_oof_fusion_v3', 'mode': 'stacking' if l2 else mode, 'meta_l2': l2,
                      'seed': seed, 'fit_split': 'train', 'meta_fit_predictions': 'question-grouped 5-fold TRAIN OOF',
                      'base_feature_names': names, 'meta_feature_names': FEATURE_NAMES, 'protocol': PROTOCOL,
                      'selection': 'argmax score, first occurrence on exact tie'}
            save(checkpoint / 'config.json', config)
            with zipfile.ZipFile(checkpoint / 'model.npz') as archive:
                unpacked = sum(i.file_size for i in archive.infolist()) + (checkpoint / 'config.json').stat().st_size
            assert unpacked < 16 * 2**20
            all_models.append({'mode': mode, 'seed': seed, 'checkpoint': str(checkpoint.relative_to(ROOT)),
                               'inference_uncompressed_bytes': unpacked,
                               'checkpoint_sha256': {name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
                                                      for name in ('model.npz', 'config.json')},
                               'full_mlp_source': str(historical.relative_to(ROOT))})
    save(destination / 'training_complete.json', {
        'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'total_training_wall_seconds': time.perf_counter() - started,
        'new_base_fits': base_records, 'meta_fits': meta_records, 'models': all_models,
        'training_questions': len(bank), 'unique_normalized_questions': len({' '.join(b['question'].split()) for b in bank}),
        'opened_jsonl': sorted(opened), 'no_validation_or_test_opened': True,
        'oof_predictions_sha256': hashlib.sha256((destination / 'train_oof_predictions.npz').read_bytes()).hexdigest(),
        'note': 'OOF used to train stacking, not an unbiased score of the complete stacking system'})
    print(json.dumps({'training_finished': True, 'new_base_fits': len(base_records), 'meta_fits': len(meta_records)}), flush=True)


if __name__ == '__main__':
    main()
