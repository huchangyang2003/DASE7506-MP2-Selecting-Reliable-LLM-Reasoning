"""Fit regularized logistic regression with NumPy L-BFGS, on train only.

Loss = mean binary cross-entropy over ALL candidates + l2/2 * ||w||^2.
Each correct candidate is a positive label. The intercept is not penalized.
This entry point has no validation/test split option and reads train only.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

# Set before importing NumPy; no parallel Python workers are created.
for thread_key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[thread_key] = '4'

import numpy as np

from common import DATA, PROTOCOL, load_inputs, parse_answer, read_jsonl


def loss_gradient(theta, features, labels, l2):
    weights, bias = theta[:-1], theta[-1]
    logits = features @ weights + bias
    residual = np.exp(-np.logaddexp(0.0, -logits)) - labels
    loss = np.mean(np.logaddexp(0.0, logits) - labels * logits)
    loss += 0.5 * l2 * (weights @ weights)
    gradient = np.empty_like(theta)
    gradient[:-1] = features.T @ residual / len(labels) + l2 * weights
    gradient[-1] = residual.mean()
    return float(loss), gradient


def fit_logistic(features, labels, l2, max_iterations=1000, gradient_tolerance=1e-6):
    theta = np.zeros(features.shape[1] + 1, dtype=np.float64)
    positive_rate = float(np.clip(labels.mean(), 1e-8, 1 - 1e-8))
    theta[-1] = np.log(positive_rate / (1 - positive_rate))
    loss, gradient = loss_gradient(theta, features, labels, l2)
    memory, trace = [], []
    reason = 'max_iterations'
    for iteration in range(max_iterations + 1):
        gradient_inf = float(np.max(np.abs(gradient)))
        trace.append({'iteration': iteration, 'loss': loss, 'gradient_inf': gradient_inf})
        if gradient_inf <= gradient_tolerance:
            reason = 'gradient_tolerance'
            break
        if iteration == max_iterations:
            break
        # L-BFGS inverse-Hessian product using the most recent ten updates.
        direction = gradient.copy()
        alphas = []
        for step, change, rho in reversed(memory):
            alpha = rho * (step @ direction)
            alphas.append(alpha)
            direction -= alpha * change
        if memory:
            step, change, _ = memory[-1]
            direction *= (step @ change) / (change @ change)
        for (step, change, rho), alpha in zip(memory, reversed(alphas)):
            direction += step * (alpha - rho * (change @ direction))
        direction *= -1
        slope = float(gradient @ direction)
        if slope >= 0:
            direction = -gradient
            slope = -float(gradient @ gradient)
            memory.clear()
        step_length = 1.0
        for _ in range(40):
            proposed = theta + step_length * direction
            new_loss, new_gradient = loss_gradient(proposed, features, labels, l2)
            if np.isfinite(new_loss) and new_loss <= loss + 1e-4 * step_length * slope:
                break
            step_length *= 0.5
        else:
            reason = 'line_search_failed'
            break
        step = proposed - theta
        change = new_gradient - gradient
        curvature = float(step @ change)
        if curvature > 1e-12:
            memory.append((step, change, 1.0 / curvature))
            memory = memory[-10:]
        theta, loss, gradient = proposed, new_loss, new_gradient
    return theta, {'converged': reason == 'gradient_tolerance', 'stop_reason': reason,
                   'iterations': len(trace) - 1, 'final_loss': loss,
                   'gradient_inf': float(np.max(np.abs(gradient))), 'trace': trace}


def training_data():
    bank, arrays = load_inputs('train')
    references = read_jsonl(DATA / 'gsm8k_train.jsonl')
    if len(references) != len(bank) or any(
            ref['id'] != row['id'] or ref['question'] != row['question']
            for ref, row in zip(references, bank)):
        raise ValueError('Training reference alignment mismatch')
    labels = []
    for row, ref in zip(bank, references):
        gold = parse_answer(ref['answer'])
        row_labels = [int((answer := parse_answer(candidate['text'])) is not None
                          and answer == gold) for candidate in row['candidates']]
        if not any(row_labels):
            raise ValueError('Training group without a correct parsed answer')
        labels.extend(row_labels)
    features = np.asarray(arrays['embeddings'], dtype=np.float64).reshape(-1, arrays['embeddings'].shape[-1])
    return features, np.asarray(labels, dtype=np.float64), len(bank)


def asset_sizes(checkpoint):
    files = [checkpoint / 'model.npz', checkpoint / 'config.json']
    with zipfile.ZipFile(files[0]) as archive:
        unpacked = sum(entry.file_size for entry in archive.infolist()) + files[1].stat().st_size
    return {'disk_bytes': sum(path.stat().st_size for path in files),
            'uncompressed_bytes': unpacked,
            'sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--l2', type=float, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.l2) or args.l2 <= 0:
        raise ValueError('l2 must be positive and finite')
    tick = time.perf_counter()
    features, labels, question_count = training_data()
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    features -= mean
    features /= scale
    optimization_tick = time.perf_counter()
    theta, optimization = fit_logistic(features, labels, args.l2)
    optimization_seconds = time.perf_counter() - optimization_tick
    if not optimization['converged']:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'failed-training.json').write_text(json.dumps({
            'l2': args.l2, 'optimization_seconds': optimization_seconds,
            'optimization': optimization}, indent=2) + '\n', encoding='utf-8')
        raise RuntimeError(f'Optimization did not converge: {optimization["stop_reason"]}, '
                           f'gradient={optimization["gradient_inf"]}')
    checkpoint = args.output_dir / 'checkpoint'
    checkpoint.mkdir(parents=True, exist_ok=True)
    np.savez(checkpoint / 'model.npz', mean=mean, scale=scale,
             weights=theta[:-1], bias=np.asarray(theta[-1]))
    config = {'method': 'embedding_logistic_v1', 'feature_dimension': features.shape[1],
              'l2': args.l2, 'standardization': 'train-candidate mean and population std; std<1e-8 becomes 1',
              'loss': 'mean_candidate_binary_cross_entropy_plus_l2_half_squared_weights',
              'selection': 'argmax candidate logit; first occurrence on exact ties',
              'training_split': 'train', 'optimizer': 'deterministic_numpy_lbfgs',
              'max_iterations': 1000, 'gradient_tolerance': 1e-6,
              'protocol': PROTOCOL}
    (checkpoint / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    sizes = asset_sizes(checkpoint)
    if max(sizes['disk_bytes'], sizes['uncompressed_bytes']) > 16 * 2**20:
        raise ValueError('Checkpoint exceeds inference asset limit')
    result = {'l2': args.l2, 'training_questions': question_count,
              'training_candidates': len(labels), 'positive_labels': int(labels.sum()),
              'optimization_seconds': optimization_seconds,
              'training_seconds': time.perf_counter() - tick,
              'assets': sizes, 'optimization': optimization,
              'seed': None, 'seed_note': 'No random initialization, shuffle, or stochastic operations',
              'test_or_validation_references_used_for_fit': False}
    result['source_sha256'] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ('train_linear.py', 'methods/embedding_linear.py', 'common.py')}
    result['train_data_sha256'] = {
        name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        for name in ('gsm8k_train.jsonl', 'candidates/gsm8k_train.jsonl', 'candidates/gsm8k_train.npz')}
    (args.output_dir / 'training.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'optimization'}))


if __name__ == '__main__':
    main()
