"""TRAIN-input-only interface verification of the already selected pooling artifact."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import stage_a  # Installs the TRAIN-only access guard; does not execute the experiment.
import numpy as np
from common import load_inputs
from pooling_core import answers, choose
from pooling_submission import predict
from submission import candidate_scores


def main():
    root = stage_a.ROOT
    run = root / 'runs/pooling/stage_a'
    output = run / (sys.argv[1] if len(sys.argv) == 2 else 'deployment_checks.json')
    if output.parent != run or output.suffix != '.json':
        raise ValueError('Use a JSON receipt filename inside the same stageA run')
    if output.exists():
        raise FileExistsError(output)
    summary = stage_a.read(run / 'summary.json')
    assert summary['status'] == 'completed' and summary['new_validation_candidate']
    cp = root / summary['selected_checkpoint']
    before = {str(p): stage_a.sha(p) for p in cp.rglob('*') if p.is_file()}
    bank, arrays = load_inputs('train', stage_a.DATA)
    bank = bank[:32]
    arrays = {k: v[:32] for k, v in arrays.items()}
    z = candidate_scores(bank, arrays, stage_a.OLD_RUN / stage_a.NAME / 'checkpoint')
    copied = candidate_scores(bank, arrays, cp / 'baseline')
    assert np.array_equal(z, copied)
    expected = choose(bank, z, summary['selection']['selected_alpha'], arrays['embeddings'])[0]
    selected = predict(bank, arrays, cp)
    assert selected == expected and all(type(x) is int and 0 <= x < 4 for x in selected)
    for count in (1, 3, 7, 32):
        assert predict(bank[:count], {k: v[:count] for k, v in arrays.items()}, cp) == selected[:count]
    assert predict([], {k: v[:0] for k, v in arrays.items()}, cp) == []
    changed = deepcopy(bank)
    for i, row in enumerate(changed):
        row['id'] = f'pooling-deployment-new-id-{i}'
    assert predict(changed, arrays, cp) == selected
    perm = np.array([2, 0, 3, 1])
    for row in changed:
        row['candidates'] = [{**row['candidates'][j], 'index': k} for k, j in enumerate(perm)]
    perm_selected = predict(changed, {k: v[:, perm] for k, v in arrays.items()}, cp)
    a, b = answers(bank), answers(changed)
    assert [a[i][j] for i, j in enumerate(selected)] == [b[i][j] for i, j in enumerate(perm_selected)]
    assert {path: stage_a.sha(path) for path in before} == before
    assert not stage_a.DENIED
    stage_a.save(output, {'passed': True, 'scope': '32 train input questions; no references loaded',
        'checkpoint_unchanged': True, 'copied_ce_logits_exact': True,
        'wrapper_matches_pure_pooling': True, 'new_ids': True, 'empty': True,
        'batches': [1, 3, 7, 32], 'permutation_answers': True,
        'new_fits': 0, 'validation_scoring_calls': 0, 'test_calls': 0,
        'bank_accesses': sorted(p for p in stage_a.OPENED if Path(p).name.startswith('gsm8k_')),
        'checkpoint_sha256': before, 'denied_accesses': stage_a.DENIED,
        'source_sha256': stage_a.sha(__file__),
        'thread_environment_set_before_numpy_import': True})
    print(json.dumps({'deployment_checks': 'passed', 'train_inputs': 32, 'validation_calls': 0}))


if __name__ == '__main__':
    main()
