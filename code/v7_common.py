"""Input-only helpers; no training, references, or research-directory dependencies."""
import json
from pathlib import Path
import numpy as np
from common import parse_answer
PUB = Path(__file__).resolve().parents[1]  # Compatibility constant; not used by inference.

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def load_asset(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def answer_information(bank):
    answers=[[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    counts=[{v:row.count(v) for v in row if v is not None} for row in answers]
    votes=np.asarray([[counts[i].get(v,0) if v is not None else 0 for v in row]
                     for i,row in enumerate(answers)],float).reshape(-1,4)
    return answers,counts,votes
