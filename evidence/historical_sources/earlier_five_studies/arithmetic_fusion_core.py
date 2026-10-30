"""Append centered arithmetic evidence to the frozen stage-3 OOF schema."""
import numpy as np
from fusion_core import FEATURE_NAMES, meta_features
from arithmetic_features import COVERAGE_NAMES, CONSISTENCY_NAMES, build_arithmetic_features

MODES = ('control', 'coverage', 'arithmetic')


def names_for(mode):
    if mode not in MODES: raise ValueError('Unknown arithmetic fusion mode')
    return FEATURE_NAMES + ([] if mode == 'control' else COVERAGE_NAMES) + (CONSISTENCY_NAMES if mode == 'arithmetic' else [])


def augmented_features(bank, mlp_scores, embedding_scores, mode, arithmetic=None):
    base = meta_features(bank, mlp_scores, embedding_scores)
    if mode == 'control': return base
    if mode not in MODES: raise ValueError('Unknown arithmetic fusion mode')
    if arithmetic is None:
        arithmetic, _ = build_arithmetic_features(bank, 'arithmetic')
    width = len(COVERAGE_NAMES) if mode == 'coverage' else len(COVERAGE_NAMES) + len(CONSISTENCY_NAMES)
    extra = arithmetic[:, :, :width]
    extra = extra - extra.mean(axis=1, keepdims=True)
    return np.concatenate([base, extra], axis=2)
