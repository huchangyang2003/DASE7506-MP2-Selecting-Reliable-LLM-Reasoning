"""Frozen base scores plus learned soft arithmetic evidence, input-only."""
import json
from pathlib import Path
import numpy as np
from features import build_features
from fusion_core import mlp_logits
from arithmetic_fusion_core import names_for, augmented_features


def candidate_scores(bank, arrays, checkpoint):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    features, names = build_features(bank, arrays, 'group')
    if names != config['base_feature_names'] or config['meta_feature_names'] != names_for(config['mode']):
        raise ValueError('Feature schema mismatch')
    with np.load(checkpoint / 'model.npz', allow_pickle=False) as m:
        mlp = mlp_logits(features, m['mean'], m['scale'], {k: m[k] for k in ('W1', 'b1', 'W2', 'b2')})
        linear = ((np.asarray(arrays['embeddings'], float) - m['emb_mean']) / m['emb_scale']) @ m['emb_weights'] + m['emb_bias'].item()
        meta = augmented_features(bank, mlp, linear, config['mode'])
        meta = (meta - m['meta_mean']) / m['meta_scale']
        return meta @ m['meta_weights'] + m['meta_bias'].item()


def predict(bank, arrays, checkpoint):
    scores = candidate_scores(bank, arrays, checkpoint)
    if scores.shape != (len(bank), 4) or not np.isfinite(scores).all():
        raise ValueError('Invalid arithmetic fusion scores')
    return [int(i) for i in scores.argmax(axis=1)]
