"""Train-only weighted BCE; constant batches, initialization and input features."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'

import numpy as np

from common import DATA, PROTOCOL, load_inputs, parse_answer, read_jsonl
from features import build_features
from no_test_access import install_guard
from weighted_bce import MODES, disagreement_mask, question_weights, weighted_network_loss_and_gradients


def fit(features, labels, weights, seed):
    rng = np.random.default_rng(seed)
    params = {'W1': (rng.standard_normal((features.shape[-1], 32)) / np.sqrt(features.shape[-1])).astype(np.float32),
              'b1': np.zeros(32, dtype=np.float32),
              'W2': (rng.standard_normal(32) / np.sqrt(32)).astype(np.float32),
              'b2': np.zeros((), dtype=np.float32)}
    initial_hash = hashlib.sha256(b''.join(value.tobytes() for value in params.values())).hexdigest()
    first = {name: np.zeros_like(value) for name, value in params.items()}
    second = {name: np.zeros_like(value) for name, value in params.items()}
    schedule = hashlib.sha256()
    step, trace = 0, []
    for epoch in range(60):
        order = rng.permutation(len(labels))
        schedule.update(order.tobytes())
        accumulated = 0.0
        for start in range(0, len(order), 128):
            ids = order[start:start + 128]
            _, gradients, data_loss = weighted_network_loss_and_gradients(params, features[ids], labels[ids], weights[ids])
            accumulated += data_loss * len(ids)
            step += 1
            for name in params:
                first[name] *= 0.9
                first[name] += 0.1 * gradients[name]
                second[name] *= 0.999
                second[name] += 0.001 * gradients[name] ** 2
                params[name] -= 0.001 * (first[name] / (1 - 0.9 ** step)) / (
                    np.sqrt(second[name] / (1 - 0.999 ** step)) + 1e-8)
        trace.append({'epoch': epoch + 1, 'online_weighted_question_data_loss': accumulated / len(labels)})
    if not all(np.isfinite(value).all() for value in params.values()):
        raise ValueError('Nonfinite model parameters')
    return params, {'initial_parameter_sha256': initial_hash, 'question_shuffle_sha256': schedule.hexdigest(),
                    'epochs': 60, 'question_batch_size': 128, 'candidates_per_full_batch': 512,
                    'hidden': 32, 'learning_rate': 0.001, 'weight_l2': 1e-4,
                    'optimizer': 'Adam beta1=0.9 beta2=0.999 epsilon=1e-8',
                    'updates': step, 'trace': trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weight-mode', choices=MODES, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    accessed = install_guard()
    tick = time.perf_counter()
    bank, arrays = load_inputs('train')
    references = read_jsonl(DATA / 'gsm8k_train.jsonl')
    if len(bank) != len(references) or any(row['id'] != ref['id'] or row['question'] != ref['question']
                                         for row, ref in zip(bank, references)):
        raise ValueError('Training alignment mismatch')
    labels = np.asarray([[int((answer := parse_answer(candidate['text'])) is not None
                             and answer == parse_answer(ref['answer']))
                          for candidate in row['candidates']] for row, ref in zip(bank, references)], dtype=np.float32)
    if not labels.any(axis=1).all():
        raise ValueError('No positive label in a group')
    weights, weight_metadata = question_weights(disagreement_mask(bank), args.weight_mode)
    features, names = build_features(bank, arrays, 'group')
    flat = features.reshape(-1, features.shape[-1])
    mean, scale = flat.mean(axis=0), flat.std(axis=0)
    scale[scale < 1e-8] = 1
    features = np.clip((features - mean) / scale, -8, 8).astype(np.float32)
    params, training = fit(features, labels, weights, args.seed)
    checkpoint = args.output_dir / 'checkpoint'
    checkpoint.mkdir(parents=True, exist_ok=True)
    np.savez(checkpoint / 'model.npz', mean=mean, scale=scale, **params)
    config = {'method': 'weighted_bce_v2', 'model_type': 'mlp', 'variant': 'group',
              'objective': 'weighted_bce', 'training_weight_mode': args.weight_mode,
              'feature_names': names, 'feature_dimension': len(names), 'hidden': 32, 'seed': args.seed,
              'fit_split': 'train', 'protocol': PROTOCOL, 'selection': 'argmax logit, first on exact tie'}
    (checkpoint / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    with zipfile.ZipFile(checkpoint / 'model.npz') as archive:
        assets = sum(entry.file_size for entry in archive.infolist()) + (checkpoint / 'config.json').stat().st_size
    assert assets < 16 * 2**20
    result = {'weight_mode': args.weight_mode, 'seed': args.seed, 'training_questions': len(bank),
              'training_seconds': time.perf_counter() - tick, 'training': training,
              'question_weights': weight_metadata, 'inference_uncompressed_bytes': assets,
              'opened_training_jsonl': sorted(accessed), 'no_validation_or_test_fit': True,
              'all_train_inputs_used_for_standardization': True,
              'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ('train_weighted.py', 'weighted_bce.py', 'objectives.py', 'features.py', 'methods/hybrid_model.py')},
              'checkpoint_sha256': {name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
                                    for name in ('model.npz', 'config.json')}}
    (args.output_dir / 'training.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({name: value for name, value in result.items() if name != 'training'}))


if __name__ == '__main__':
    main()
