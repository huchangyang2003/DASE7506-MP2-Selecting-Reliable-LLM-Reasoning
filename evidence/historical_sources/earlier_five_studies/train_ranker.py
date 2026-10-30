"""Train matched MLP32 models; the only experiment variable is the objective."""

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
from objectives import network_loss_and_gradients


def fit(features, labels, objective, seed):
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
            _, gradients, data_loss = network_loss_and_gradients(params, features[ids], labels[ids], objective)
            accumulated += data_loss * len(ids)
            step += 1
            for name in params:
                first[name] *= 0.9
                first[name] += 0.1 * gradients[name]
                second[name] *= 0.999
                second[name] += 0.001 * gradients[name] ** 2
                params[name] -= 0.001 * (first[name] / (1 - 0.9 ** step)) / (
                    np.sqrt(second[name] / (1 - 0.999 ** step)) + 1e-8)
        trace.append({'epoch': epoch + 1, 'online_mean_question_data_loss': accumulated / len(labels)})
    if not all(np.isfinite(value).all() for value in params.values()):
        raise ValueError('Nonfinite model parameters')
    return params, {'initial_parameter_sha256': initial_hash, 'question_shuffle_sha256': schedule.hexdigest(),
                    'epochs': 60, 'question_batch_size': 128, 'candidates_per_full_batch': 512,
                    'hidden': 32, 'learning_rate': 0.001, 'weight_l2': 1e-4,
                    'optimizer': 'Adam beta1=0.9 beta2=0.999 epsilon=1e-8',
                    'updates': step, 'trace': trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objective', choices=('bce', 'pairwise', 'listwise'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    accessed = install_guard()
    tick = time.perf_counter()
    bank, arrays = load_inputs('train')
    references = read_jsonl(DATA / 'gsm8k_train.jsonl')
    if len(bank) != len(references) or any(row['id'] != ref['id'] or row['question'] != ref['question']
                                         for row, ref in zip(bank, references)):
        raise ValueError('Training reference alignment mismatch')
    labels = np.asarray([[int((answer := parse_answer(candidate['text'])) is not None
                             and answer == parse_answer(ref['answer']))
                          for candidate in row['candidates']] for row, ref in zip(bank, references)], dtype=np.float32)
    if not labels.any(axis=1).all():
        raise ValueError('Group without a positive label')
    features, names = build_features(bank, arrays, 'group')
    flat = features.reshape(-1, features.shape[-1])
    mean, scale = flat.mean(axis=0), flat.std(axis=0)
    scale[scale < 1e-8] = 1
    features = np.clip((features - mean) / scale, -8, 8).astype(np.float32)
    params, training = fit(features, labels, args.objective, args.seed)
    checkpoint = args.output_dir / 'checkpoint'
    checkpoint.mkdir(parents=True, exist_ok=True)
    np.savez(checkpoint / 'model.npz', mean=mean, scale=scale, **params)
    config = {'method': 'objective_comparison_v2', 'model_type': 'mlp', 'variant': 'group',
              'objective': args.objective, 'feature_names': names, 'feature_dimension': len(names),
              'hidden': 32, 'seed': args.seed, 'fit_split': 'train', 'protocol': PROTOCOL,
              'selection': 'argmax logit, first occurrence on exact tie'}
    (checkpoint / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    with zipfile.ZipFile(checkpoint / 'model.npz') as archive:
        assets = sum(entry.file_size for entry in archive.infolist()) + (checkpoint / 'config.json').stat().st_size
    assert assets < 16 * 2**20
    result = {'objective': args.objective, 'seed': args.seed, 'training_questions': len(bank),
              'positive_candidates': int(labels.sum()), 'all_four_correct_groups': int(np.sum(labels.sum(axis=1) == 4)),
              'training_seconds': time.perf_counter() - tick, 'training': training,
              'inference_uncompressed_bytes': assets, 'opened_training_jsonl': sorted(accessed),
              'no_validation_or_test_fit': True,
              'checkpoint_sha256': {name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
                                    for name in ('model.npz', 'config.json')},
              'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ('train_ranker.py', 'objectives.py', 'features.py', 'methods/hybrid_model.py')}}
    (args.output_dir / 'training.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'training'}))


if __name__ == '__main__':
    main()
