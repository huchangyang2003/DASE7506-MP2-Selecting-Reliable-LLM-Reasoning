"""Frozen inference, with no reference-label access or question-ID lookup."""
import json
from pathlib import Path
import numpy as np
from common import parse_answer

def candidate_scores(bank,arrays,checkpoint):
    cp=Path(checkpoint);cfg=json.loads((cp/"config.json").read_text(encoding="utf-8"))
    if not bank:return np.empty((0,4),dtype=float)
    if cfg["method"]=="reference_full":
        import reference_features as rf
        rf.parse_answer=parse_answer
        x=np.concatenate(rf.build_features(bank,arrays),axis=-1)
        with np.load(cp/"model.npz",allow_pickle=False) as z:
            t=np.clip((x.astype(np.float64)-z["mean"])/z["scale"],-20.,20.)
            return np.mean([t@w for w in z["weights"]],axis=0)
    if cfg["method"]=="tree":
        import lightgbm as lgb
        from linear_core import build_inputs
        x,names=build_inputs(bank,arrays);x=x[:,:,1024:]
        assert names[1024:]==cfg["hand_names"]
        if cfg["pca32"]:
            with np.load(cp/"transform.npz",allow_pickle=False) as z:
                emb=(arrays["embeddings"].astype(np.float64)-z["pca_mean"])@z["pca_axes"]
            x=np.concatenate([x,emb],axis=-1)
        tree=lgb.Booster(model_file=str(cp/"tree.txt"))
        return tree.predict(np.ascontiguousarray(x.reshape(-1,x.shape[-1])),num_threads=4).reshape(x.shape[:2])
    raise ValueError("Unknown frozen method")

def predict(bank,arrays,checkpoint):
    if not bank:return []
    return [int(i) for i in candidate_scores(bank,arrays,checkpoint).argmax(axis=1)]
