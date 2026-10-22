"""Guarded wrapper around unchanged official validation commands."""

import runpy
import sys

from no_test_access import install_guard

if __name__ == '__main__':
    install_guard()
    sys.argv = sys.argv[1:]
    if sys.argv[0] not in ('predict.py', 'score_predictions.py'):
        raise ValueError('Only the official runner and scorer are supported')
    if '--split' not in sys.argv or sys.argv[sys.argv.index('--split') + 1] != 'validation':
        raise ValueError('Only explicit validation scoring is allowed in this stage')
    runpy.run_path(sys.argv[0], run_name='__main__')
