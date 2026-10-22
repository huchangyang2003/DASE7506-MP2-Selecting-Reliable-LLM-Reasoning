"""Check a supplied prediction module under a checkpoint-only file/network guard."""
import copy
from pathlib import Path
import sys
import numpy as np
import numpy.testing
from common import parse_answer


def verify(method, bank, arrays, checkpoint, official):
    checkpoint = Path(checkpoint).resolve()
    official = np.asarray(official)
    allowed = {checkpoint / n for n in ('model.npz', 'config.json')}
    active = [False]; opened = []
    def audit(event, args):
        if not active[0]: return
        if event.startswith('socket.'): raise PermissionError('Network disabled during inference')
        if event == 'open' and isinstance(args[0], str):
            p = Path(args[0]).resolve()
            if p not in allowed: raise PermissionError(f'Non-checkpoint inference read: {p}')
            opened.append(str(p))
    sys.addaudithook(audit)
    active[0] = True
    try:
        choices = method.predict(bank, arrays, checkpoint)
        assert choices == official.tolist()
        short = copy.deepcopy(bank[:17])
        for i, row in enumerate(short): row['id'] = f'unseen_arithmetic_id_{i}'
        assert method.predict(short, {k: a[:17] for k, a in arrays.items()}, checkpoint) == choices[:17]
        permutation = np.array([2, 0, 3, 1])
        reordered = copy.deepcopy(bank)
        for row in reordered:
            row['candidates'] = [{**row['candidates'][j], 'index': i} for i, j in enumerate(permutation)]
        scores = method.candidate_scores(bank, arrays, checkpoint)
        reordered_scores = method.candidate_scores(reordered, {k: a[:, permutation] for k, a in arrays.items()}, checkpoint)
        np.testing.assert_allclose(reordered_scores, scores[:, permutation], rtol=1e-5, atol=1e-5)
        mapped = permutation[reordered_scores.argmax(axis=1)]
        changes = np.flatnonzero(mapped != official)
        change_details = [{'id': bank[q]['id'], 'original_index': int(official[q]), 'mapped_index': int(mapped[q]),
                           'exact_tie': bool(scores[q, official[q]] == scores[q, mapped[q]]),
                           'same_parsed_answer': parse_answer(bank[q]['candidates'][official[q]]['text']) ==
                                                 parse_answer(bank[q]['candidates'][mapped[q]]['text'])} for q in changes]
    finally:
        active[0] = False
    return {'reload_all_choices_match': True, 'new_ids_17_rows_match': True,
            'permuted_scores_allclose': True, 'permutation_index_changes': len(changes),
            'permutation_parsed_answer_changes': sum(not r['same_parsed_answer'] for r in change_details),
            'changed_choices': change_details, 'checkpoint_only_read_files': sorted(set(opened)),
            'network_guard_enabled': True, 'no_dataset_reads_during_prediction': True}
