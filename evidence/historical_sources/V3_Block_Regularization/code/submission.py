"""Input-only linear selector; no reference data, IDs or split conditions."""
from pathlib import Path
import json
import numpy as np
from linear_core import build_inputs, scores


def candidate_scores(bank, arrays, checkpoint):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint/"config.json").read_text(encoding="utf-8"))
    x, names = build_inputs(bank, arrays)
    if names != config["feature_names"]:
        raise ValueError("Feature schema mismatch")
    with np.load(checkpoint/"model.npz", allow_pickle=False) as z:
        result = scores(x, {k:z[k] for k in ("mean","scale","weights","bias")})
    if result.shape != (len(bank),4) or not np.isfinite(result).all():
        raise ValueError("Invalid scores")
    return result


def predict(bank, arrays, checkpoint):
    return candidate_scores(bank, arrays, checkpoint).argmax(axis=1).astype(int).tolist()
