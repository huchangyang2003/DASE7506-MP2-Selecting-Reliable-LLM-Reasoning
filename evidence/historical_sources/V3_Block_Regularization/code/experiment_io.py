"""Local provenance, official split loading and audit guards."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np
from common import load_inputs, read_jsonl, parse_answer

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent
DATA = PROJECT / "DASE7506-MP2-main/code/data"
FROZEN = PROJECT / "Final_Optimized_Solution"
PUBLICATION = PROJECT / "First_Publication_20261022/DASE7506-MP2-Selecting-Reliable-LLM-Reasoning"


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard(role):
    if role not in ("train", "validation_predict", "validation_score", "analysis"):
        raise ValueError(role)
    opened, denied = set(), []
    def audit(event, args):
        if event.startswith("socket.connect"):
            raise PermissionError("This experiment is offline")
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        p = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
        name = p.name.lower()
        blocked = name.startswith(("gsm8k_test", "gsm8k_audit", "test-predictions", "test-score"))
        if role == "train" and name.startswith(("gsm8k_validation", "validation-predictions", "validation-score")):
            blocked = True
        if role == "validation_predict" and name == "gsm8k_validation.jsonl" and p.parent.name != "candidates":
            blocked = True
        if blocked:
            denied.append(str(p))
            raise PermissionError("Forbidden split/reference access in " + role)
        if name.endswith(".jsonl") or name.startswith("gsm8k_"):
            opened.add(str(p.resolve()))
    sys.addaudithook(audit)
    return opened, denied


def load_split(split):
    if split not in ("train", "validation"):
        raise ValueError("Only train and validation are allowed")
    bank, arrays = load_inputs(split, DATA)
    refs = read_jsonl(DATA / ("gsm8k_"+split+".jsonl"))
    if len(bank) != len(refs) or any(b["id"] != r["id"] or b["question"] != r["question"] for b,r in zip(bank,refs)):
        raise ValueError("Reference alignment")
    y = np.asarray([[int((a := parse_answer(c["text"])) is not None and a == parse_answer(r["answer"]))
                    for c in b["candidates"]] for b,r in zip(bank,refs)], dtype=float)
    if not np.all(y.any(axis=1)):
        raise ValueError("No positive candidate")
    return bank, arrays, y


def disagreement(bank):
    return np.asarray([len({a for c in b["candidates"] if (a := parse_answer(c["text"])) is not None}) >= 2
                       for b in bank], dtype=bool)


def summary(y, selected, hard):
    correct = y[np.arange(len(y)), selected].astype(bool)
    return {**metrics(correct, hard), "selected": selected.tolist()}


def metrics(correct, hard):
    def item(mask):
        n, c = int(mask.sum()), int(correct[mask].sum())
        return {"n": n, "correct": c, "accuracy": 100*c/n if n else None}
    return {"all": item(np.ones(len(correct), dtype=bool)),
            "disagreement": item(hard), "other": item(~hard)}


def snapshot_original():
    frozen = read(FROZEN / "FREEZE.json")
    for rel, metadata in frozen["files"].items():
        assert sha(FROZEN/rel) == metadata["sha256"], rel
    pub = read(PUBLICATION / "PUBLICATION_MANIFEST.json")
    expected = pub["files"]
    return {"frozen_files": {rel: sha(FROZEN/rel) for rel in frozen["files"]},
            "frozen_selection": sha(FROZEN/"FINAL_SELECTION.json"),
            "frozen_report": sha(FROZEN/"output/pdf/DASE7506_MP2_Final_Report.pdf"),
            "publication_manifest": sha(PUBLICATION/"PUBLICATION_MANIFEST.json"),
            "publication_expected_files": len(expected),
            "publication_head": (PUBLICATION/".git/refs/heads/main").read_text().strip()}


def source_snapshot():
    return {p.name: sha(p) for p in sorted((ROOT/"code").glob("*.py"))}
