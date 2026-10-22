"""Candidate BCE and two multi-positive, question-level ranking objectives."""

import numpy as np


def _logsumexp(values):
    maximum = values.max(axis=1, keepdims=True)
    return maximum[:, 0] + np.log(np.exp(values - maximum).sum(axis=1))


def loss_and_gradient(logits, labels, objective):
    """Return mean question loss and d(loss)/d(logits), shape [B,4].

    Each ranking question with four positive labels contributes exactly zero.
    Pairwise loss averages positive-negative pairs WITHIN each question, then
    averages questions (zero-loss questions remain in the denominator).
    Listwise loss is -log(sum of softmax mass on ANY correct candidate).
    """
    logits, labels = np.asarray(logits), np.asarray(labels)
    if (logits.ndim != 2 or logits.shape != labels.shape or logits.shape[1] != 4
            or not len(logits) or not np.isfinite(logits).all()
            or not np.isin(labels, [0, 1]).all() or not labels.any(axis=1).all()):
        raise ValueError('Expected finite [B,4] logits and binary labels with at least one positive per group')
    n = len(logits)
    if objective == 'bce':
        loss = np.mean(np.logaddexp(0, logits) - labels * logits)
        gradient = (np.exp(-np.logaddexp(0, -logits)) - labels) / logits.size
    elif objective == 'pairwise':
        # Axis 1 is a positive candidate, axis 2 a negative candidate.
        mask = labels[:, :, None] * (1 - labels[:, None, :])
        weights = mask / np.maximum(mask.sum(axis=(1, 2), keepdims=True), 1)
        delta = logits[:, None, :] - logits[:, :, None]
        loss = np.mean(np.sum(np.logaddexp(0, delta) * weights, axis=(1, 2)))
        residual = np.exp(-np.logaddexp(0, -delta)) * weights / n
        gradient = residual.sum(axis=1) - residual.sum(axis=2)
    elif objective == 'listwise':
        positives = np.where(labels > 0, logits, -np.inf)
        log_total, log_positive = _logsumexp(logits), _logsumexp(positives)
        loss = np.mean(log_total - log_positive)
        gradient = (np.exp(logits - log_total[:, None])
                    - np.exp(positives - log_positive[:, None])) / n
    else:
        raise ValueError('Unknown objective')
    return float(loss), gradient


def network_loss_and_gradients(params, features, labels, objective, l2=1e-4):
    flat = features.reshape(-1, features.shape[-1])
    preactivation = flat @ params['W1'] + params['b1']
    hidden = np.maximum(preactivation, 0)
    logits = (hidden @ params['W2'] + params['b2']).reshape(labels.shape)
    data_loss, residual = loss_and_gradient(logits, labels, objective)
    residual = residual.ravel()
    hidden_gradient = residual[:, None] * params['W2'][None, :] * (preactivation > 0)
    gradients = {'W1': flat.T @ hidden_gradient + l2 * params['W1'],
                 'b1': hidden_gradient.sum(axis=0),
                 'W2': hidden.T @ residual + l2 * params['W2'],
                 'b2': np.asarray(residual.sum())}
    regularization = 0.5 * l2 * (np.sum(params['W1'] ** 2) + np.sum(params['W2'] ** 2))
    return data_loss + float(regularization), gradients, data_loss
