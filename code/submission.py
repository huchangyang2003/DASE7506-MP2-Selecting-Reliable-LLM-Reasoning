"""Frozen CE inference followed by a fixed planned pooling strength."""
from pathlib import Path
import json
from pooling_core import choose

from ce_core import candidate_scores


def predict(bank, arrays, checkpoint):
    root = Path(checkpoint)
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    if config['method'] != 'v6_ce_answer_pooling' or config['temperature'] != 1:
        raise ValueError('Unsupported pooling configuration')
    logits = candidate_scores(bank, arrays, root / 'baseline')
    return choose(bank, logits, config['alpha'], arrays['embeddings'])[0]
