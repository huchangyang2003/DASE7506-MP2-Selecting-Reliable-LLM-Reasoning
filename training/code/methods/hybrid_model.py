"""Frozen candidate scorer with input-only engineered features."""

import json
from pathlib import Path

import numpy as np

from features import build_features


def candidate_scores(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Hybrid method requires a checkpoint')
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    features, names = build_features(bank, arrays, config['variant'])
    if names != config['feature_names']:
        raise ValueError('Checkpoint feature schema differs from source')
    with np.load(checkpoint / 'model.npz', allow_pickle=False) as model:
        features = (features - model['mean']) / model['scale']
        if config['model_type'] == 'linear':
            return features @ model['weights'] + model['bias'].item()
        if config['model_type'] == 'mlp':
            features = np.clip(features, -8, 8).astype(np.float32)
            hidden = np.maximum(features @ model['W1'] + model['b1'], 0)
            return hidden @ model['W2'] + model['b2'].item()
        raise ValueError('Unknown model type')


def predict(bank, arrays, checkpoint):
    scores = candidate_scores(bank, arrays, checkpoint)
    if scores.shape != (len(bank), 4) or not np.isfinite(scores).all():
        raise ValueError('Invalid candidate scores')
    return [int(index) for index in scores.argmax(axis=1)]
