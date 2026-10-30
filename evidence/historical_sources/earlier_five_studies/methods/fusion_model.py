"""Self-contained OOF stacking inference; never opens a dataset or references."""
import json
from pathlib import Path
import numpy as np
from features import build_features
from fusion_core import FEATURE_NAMES, meta_features, fixed_average, mlp_logits


def candidate_scores(bank, arrays, checkpoint):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    features, names = build_features(bank, arrays, 'group')
    if names != config['base_feature_names']:
        raise ValueError('Base feature schema mismatch')
    with np.load(checkpoint / 'model.npz', allow_pickle=False) as model:
        mlp = mlp_logits(features, model['mean'], model['scale'],
                         {key: model[key] for key in ('W1', 'b1', 'W2', 'b2')})
        if config['mode'] == 'mlp_control':
            return mlp
        embedding = (np.asarray(arrays['embeddings'], float) - model['emb_mean']) / model['emb_scale']
        linear = embedding @ model['emb_weights'] + model['emb_bias'].item()
        if config['mode'] == 'equal_average':
            return fixed_average(bank, mlp, linear)
        if config['mode'] != 'stacking' or config['meta_feature_names'] != FEATURE_NAMES:
            raise ValueError('Unknown fusion configuration')
        meta = (meta_features(bank, mlp, linear) - model['meta_mean']) / model['meta_scale']
        return meta @ model['meta_weights'] + model['meta_bias'].item()


def predict(bank, arrays, checkpoint):
    scores = candidate_scores(bank, arrays, checkpoint)
    if scores.shape != (len(bank), 4) or not np.isfinite(scores).all():
        raise ValueError('Invalid fused scores')
    return [int(i) for i in scores.argmax(axis=1)]
