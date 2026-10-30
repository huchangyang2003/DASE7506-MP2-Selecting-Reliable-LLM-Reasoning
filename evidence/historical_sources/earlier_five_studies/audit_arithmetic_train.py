"""Train-input-only parser coverage and inspectable examples; no validation."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import numpy as np
from common import load_bank
from arithmetic_features import build_arithmetic_features
from no_test_access import install_guard
from train_stage3 import train_only_guard


def main():
    install_guard(); sys.addaudithook(train_only_guard)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('runs/stage4_parser'))
    folder = parser.parse_args().output_dir
    if folder.exists(): raise FileExistsError('Keep the initial parser audit')
    folder.mkdir(parents=True)
    started = time.perf_counter()
    bank = load_bank('train')
    values, names, details = build_arithmetic_features(bank, details=True)
    np.savez(folder / 'train_arithmetic_features.npz', features=values)
    flat = [r for group in details for r in group]
    covered = np.asarray([r['checked'] > 0 for r in flat])
    examples = []
    # Deterministic first examples from four parser states, no reference labels consulted.
    for category in ('bad', 'rounding_only', 'exact', 'no_checked'):
        collected = 0
        for q, group in enumerate(details):
            for j, r in enumerate(group):
                matches = {'bad': r['bad'] > 0, 'rounding_only': r['accepted'] > r['exact'],
                           'exact': r['checked'] > 0 and r['exact'] == r['checked'],
                           'no_checked': r['checked'] == 0}[category]
                if matches:
                    examples.append({'category': category, 'id': bank[q]['id'], 'candidate': j,
                                     'text': bank[q]['candidates'][j]['text'], 'parser': r})
                    collected += 1
                if collected >= 8: break
            if collected >= 8: break
    (folder / 'examples.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    summary = {'training_candidates': len(flat), 'covered_candidates': int(covered.sum()),
               'coverage_percent': float(100*covered.mean()), 'equals': sum(r['equals'] for r in flat),
               'checked_equations': sum(r['checked'] for r in flat), 'exact_equations': sum(r['exact'] for r in flat),
               'accepted_equations': sum(r['accepted'] for r in flat), 'bad_equations': sum(r['bad'] for r in flat),
               'candidates_with_bad_equation': sum(r['bad'] > 0 for r in flat),
               'feature_names': names, 'parser_sha256': hashlib.sha256(Path('arithmetic_features.py').read_bytes()).hexdigest(),
               'seconds': time.perf_counter()-started, 'reference_labels_consulted': False,
               'note': 'Input-only AI-assisted parser audit, not independent human accuracy validation.'}
    (folder / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__': main()
