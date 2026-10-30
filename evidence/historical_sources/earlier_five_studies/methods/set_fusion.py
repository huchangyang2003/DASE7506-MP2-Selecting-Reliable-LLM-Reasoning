"""Input-only frozen arithmetic fusion with optional shared/set residual head."""
import json
from pathlib import Path
import numpy as np
from arithmetic_fusion_core import names_for,augmented_features
from features import build_features
from fusion_core import mlp_logits
from set_network import forward


def candidate_scores(bank,arrays,checkpoint):
    checkpoint=Path(checkpoint)
    config=json.loads((checkpoint/'config.json').read_text(encoding='utf-8'))
    features,names=build_features(bank,arrays,'group')
    if names!=config['base_feature_names'] or config['meta_feature_names']!=names_for('arithmetic'):
        raise ValueError('Frozen feature schema mismatch')
    with np.load(checkpoint/'model.npz',allow_pickle=False) as m:
        mlp=mlp_logits(features,m['mean'],m['scale'],{k:m[k] for k in ('W1','b1','W2','b2')})
        linear=((np.asarray(arrays['embeddings'],float)-m['emb_mean'])/m['emb_scale'])@m['emb_weights']+m['emb_bias'].item()
        meta=augmented_features(bank,mlp,linear,'arithmetic')
        standardized=(meta-m['meta_mean'])/m['meta_scale']
        offset=standardized@m['meta_weights']+m['meta_bias'].item()
        if config['mode']=='control':return offset
        p={name.removeprefix('joint_'):m[name] for name in m.files if name.startswith('joint_')}
        return forward(p,np.clip(standardized,-8,8).astype(np.float32),offset,config['mode'])


def predict(bank,arrays,checkpoint):
    scores=candidate_scores(bank,arrays,checkpoint)
    if scores.shape!=(len(bank),4) or not np.isfinite(scores).all():raise ValueError('Invalid set-model scores')
    return [int(i) for i in scores.argmax(axis=1)]
