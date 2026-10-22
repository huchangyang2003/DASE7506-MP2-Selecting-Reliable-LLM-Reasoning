"""Input-defined question difficulty and globally normalized weighted BCE."""

import numpy as np

from common import parse_answer
from objectives import network_loss_and_gradients

MODES = ('uniform', 'hard_x2', 'hard_x4', 'hard_only')


def disagreement_mask(bank):
    return np.array([len({answer for candidate in row['candidates']
                         if (answer := parse_answer(candidate['text'])) is not None}) >= 2
                     for row in bank], dtype=bool)


def question_weights(hard, mode):
    hard = np.asarray(hard, dtype=bool)
    if hard.ndim != 1 or not len(hard):
        raise ValueError('Expected a nonempty question mask')
    if mode == 'uniform':
        raw = np.ones(len(hard))
    elif mode in ('hard_x2', 'hard_x4'):
        raw = np.where(hard, 2.0 if mode == 'hard_x2' else 4.0, 1.0)
    elif mode == 'hard_only':
        raw = hard.astype(float)
    else:
        raise ValueError('Unknown weighting mode')
    if raw.mean() <= 0:
        raise ValueError('No weighted training questions')
    # Global TRAIN mean, not per-batch mean; preserves the expected loss scale.
    normalized = (raw / raw.mean()).astype(np.float32)
    return normalized, {'mode': mode, 'train_questions': len(hard), 'disagreement_questions': int(hard.sum()),
                         'other_questions': int((~hard).sum()), 'raw_weight_mean': float(raw.mean()),
                         'normalization': 'Divide by full TRAIN raw-weight mean; no batch renormalization',
                         'mean_normalized_weight': float(normalized.mean()),
                         'hard_weight': float(normalized[hard][0]) if hard.any() else None,
                         'other_weight': float(normalized[~hard][0]) if (~hard).any() else None,
                         'hard_fraction_of_total_weight': float(raw[hard].sum() / raw.sum())}


def weighted_network_loss_and_gradients(params, features, labels, weights, l2=1e-4):
    weights = np.asarray(weights)
    if weights.shape != (len(labels),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Expected finite nonnegative per-question weights')
    if np.all(weights == 1):
        # Preserve the exact arithmetic of the recorded stage-1 BCE control.
        return network_loss_and_gradients(params, features, labels, 'bce', l2)
    flat = features.reshape(-1, features.shape[-1])
    preactivation = flat @ params['W1'] + params['b1']
    hidden = np.maximum(preactivation, 0)
    logits = (hidden @ params['W2'] + params['b2']).reshape(labels.shape)
    per_question_bce = np.mean(np.logaddexp(0, logits) - labels * logits, axis=1)
    data_loss = float(np.mean(weights * per_question_bce))
    residual = ((np.exp(-np.logaddexp(0, -logits)) - labels)
                * weights[:, None] / logits.size).ravel()
    hidden_gradient = residual[:, None] * params['W2'][None, :] * (preactivation > 0)
    gradients = {'W1': flat.T @ hidden_gradient + l2 * params['W1'],
                 'b1': hidden_gradient.sum(axis=0),
                 'W2': hidden.T @ residual + l2 * params['W2'],
                 'b2': np.asarray(residual.sum())}
    regularization = 0.5 * l2 * (np.sum(params['W1'] ** 2) + np.sum(params['W2'] ** 2))
    return data_loss + float(regularization), gradients, data_loss
