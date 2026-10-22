"""Fail closed if a development process tries to open public test data/results."""

from pathlib import Path
import sys


def install_guard():
    opened_jsonl = set()

    def audit(event, arguments):
        if event != 'open' or not isinstance(arguments[0], str):
            return
        name = Path(arguments[0]).name.lower()
        if name.startswith(('gsm8k_test', 'test-score', 'test-predictions')):
            raise PermissionError('V2 objective development must not access test inputs, references or predictions')
        if name.endswith('.jsonl'):
            opened_jsonl.add(str(Path(arguments[0]).resolve()))

    sys.addaudithook(audit)
    return opened_jsonl
