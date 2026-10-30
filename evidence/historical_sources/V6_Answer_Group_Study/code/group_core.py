"""Fixed stage-B representations, weighted pair loss and NumPy inference.

Numeric schema/bounded token extraction adapted from CaspianW/dase7506-mp2,
commit 6709b9ff3cfaea1fed0a4f897f6796358ccfcd02, question_operation_features.py.
The L-BFGS two-loop/Armijo routine follows our frozen V3 linear_core.py.
The course final-answer parser remains unchanged.
"""
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import sys
import numpy as np

BASE_CODE = Path(__file__).resolve().parents[2] / 'V3_Block_Regularization/code'
sys.path.insert(0, str(BASE_CODE))
from common import parse_answer

MODES = ('candidate_pair', 'group_mean', 'group_mean_context')
WIDTH = 1030
LAMBDA = 0.01
VARIANCE_FLOOR = 1e-8
LIMIT = 10**12
PART = r'(?:\d[\d,]*(?:\.\d+)?|\.\d+)'
TOKEN = re.compile(r'(?<![\w.,/])[-+]?' + PART + r'(?:/[-+]?' + PART + r')?,?(?![\w,/]|\.\d)')
STRICT = re.compile(r'[-+]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?|[-+]?\.\d+')
SURFACE = ['answer_sign', 'log1p_clipped_absolute_answer', 'answer_is_integer',
           'answer_equals_question_number', 'log1p_question_number_count',
           'log1p_answer_to_question_max_magnitude']


def question_numbers(text):
    values = []
    for match in TOKEN.finditer(str(text)[:20000]):
        token = match.group().rstrip(',')
        if len(token) > 80:
            continue
        parts = token.split('/')
        if any(STRICT.fullmatch(part) is None for part in parts):
            continue
        try:
            value = Fraction(parts[0].replace(',', ''))
            if len(parts) == 2:
                value /= Fraction(parts[1].replace(',', ''))
        except (ValueError, ZeroDivisionError):
            continue
        if abs(value) <= LIMIT and value.denominator <= LIMIT:
            values.append(value)
            if len(values) == 16:
                break
    return values


def surface(answer, numbers):
    value = Fraction(answer)
    magnitude = float(min(abs(value), LIMIT))
    maximum = max((abs(x) for x in numbers), default=Fraction(0))
    divisor = float(maximum) if maximum else 1.0
    return np.asarray([float((value > 0) - (value < 0)), math.log1p(magnitude),
        float(value.denominator == 1), float(value in numbers), math.log1p(len(numbers)),
        math.log1p(magnitude / divisor)], dtype=np.float64)


def build_inputs(bank, arrays):
    emb = np.asarray(arrays['embeddings'], dtype=np.float64)
    if emb.shape != (len(bank), 4, 1024) or not np.isfinite(emb).all():
        raise ValueError('Expected finite [N,4,1024] embeddings')
    extra = np.zeros((len(bank), 4, 6), dtype=np.float64)
    parsed, groups = [], []
    for q, row in enumerate(bank):
        if len(row['candidates']) != 4:
            raise ValueError('Expected four candidates')
        values = [parse_answer(c['text']) for c in row['candidates']]
        parsed.append(values)
        nums = question_numbers(row['question'])
        buckets = {}
        for j, value in enumerate(values):
            if value is not None:
                extra[q, j] = surface(value, nums)
                buckets.setdefault(value, []).append(j)
        ordered = []
        for answer in sorted(buckets):
            members = sorted(buckets[answer], key=lambda j: (
                row['candidates'][j]['text'], emb[q, j].tobytes()))
            ordered.append((answer, members))
        groups.append(ordered)
    x = np.concatenate([emb, extra], axis=2)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite numeric surface feature')
    return x, parsed, groups


def group_values(x, groups):
    result = []
    for q, row in enumerate(groups):
        local = []
        for _, members in row:
            block = x[q, members]
            mean = block.mean(axis=0)
            spread = np.square(block[:, :1024] - mean[:1024]).mean()
            local.append(np.r_[mean, len(members) / 4.0, spread])
        result.append(local)
    return result


def scaler(values):
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid scaler fitting rows')
    mean, variance = values.mean(axis=0), values.var(axis=0)
    scale = np.sqrt(variance)
    scale[variance < VARIANCE_FLOOR] = 1.0
    return mean, scale


def fit_statistics(x, grouped, fit):
    fit = np.asarray(fit, dtype=int)
    mean, scale = scaler(x[fit].reshape(-1, WIDTH))
    original_groups = np.asarray([g[-2:] for q in fit for g in grouped[q]])
    extra_mean, extra_scale = scaler(original_groups)
    return {'mean': np.r_[mean, extra_mean], 'scale': np.r_[scale, extra_scale]}


def pair_design(x, grouped, parsed, groups, labels, fit, mode, stats):
    if mode not in MODES:
        raise ValueError(mode)
    dim = 1032 if mode == 'group_mean_context' else WIDTH
    differences, weights, owners, positives, negatives = [], [], [], [], []
    for q in fit:
        good, bad = [], []
        for g, (_, members) in enumerate(groups[q]):
            truth = np.asarray(labels[q, members], dtype=bool)
            if not np.all(truth == truth[0]):
                raise ValueError('Equivalent answers must share their label')
            (good if truth[0] else bad).append(g)
        if not good or not bad:
            continue
        group_pair_weight = 1.0 / (len(good) * len(bad))
        for p in good:
            for n in bad:
                pm, nm = groups[q][p][1], groups[q][n][1]
                if mode == 'candidate_pair':
                    for a in pm:
                        for b in nm:
                            differences.append(x[q, a] - x[q, b])
                            weights.append(group_pair_weight / (len(pm) * len(nm)))
                            owners.append(q); positives.append(1 << a); negatives.append(1 << b)
                else:
                    differences.append(grouped[q][p][:dim] - grouped[q][n][:dim])
                    weights.append(group_pair_weight)
                    owners.append(q)
                    positives.append(sum(1 << j for j in pm)); negatives.append(sum(1 << j for j in nm))
    if not differences:
        raise ValueError('No informative positive/negative answer pairs')
    X = np.asarray(differences, dtype=np.float64) / stats['scale'][:dim]
    w = np.asarray(weights, dtype=np.float64)
    owners = np.asarray(owners, dtype=int)
    totals = np.bincount(owners, weights=w, minlength=len(x))
    informative = np.flatnonzero(totals > 0)
    if not np.allclose(totals[informative], 1.0, rtol=1e-13, atol=1e-13):
        raise ValueError('Every informative question must have total weight one')
    assert set(informative) <= set(fit)
    return X, w, {'owners': owners, 'positive_mask': np.asarray(positives, dtype=np.int8),
        'negative_mask': np.asarray(negatives, dtype=np.int8), 'row_weights': w,
        'question_total_weights': totals, 'informative_questions': informative}


def objective(w, X, weights):
    normalized = weights / weights.sum()
    margin = X @ w
    value = float(normalized @ np.logaddexp(0, -margin) + 0.5 * LAMBDA * (w @ w))
    tail = np.exp(-np.logaddexp(0, margin))
    gradient = -(X.T @ (normalized * tail)) + LAMBDA * w
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite pair objective')
    return value, gradient


def fit(X, weights, maxiter=2000, gtol=1e-6):
    w = np.zeros(X.shape[1], dtype=np.float64)
    value, gradient = objective(w, X, weights)
    memory, trace, evaluations = [], [], 1
    reason = 'max_iterations'
    for iteration in range(maxiter + 1):
        norm = float(np.abs(gradient).max())
        trace.append({'iteration': iteration, 'loss': value, 'gradient_inf': norm})
        if norm <= gtol:
            reason = 'gradient_tolerance'
            break
        if iteration == maxiter:
            break
        direction, alphas = gradient.copy(), []
        for step, change, rho in reversed(memory):
            alpha = rho * np.dot(step, direction)
            alphas.append(alpha)
            direction -= alpha * change
        if memory:
            step, change, _ = memory[-1]
            direction *= np.dot(step, change) / np.dot(change, change)
        for (step, change, rho), alpha in zip(memory, reversed(alphas)):
            direction += step * (alpha - rho * np.dot(change, direction))
        direction = -direction
        slope = float(np.dot(gradient, direction))
        if slope >= 0:
            direction, slope = -gradient, -float(np.dot(gradient, gradient))
            memory.clear()
        length = 1.0
        for _ in range(40):
            proposed = w + length * direction
            new_value, new_gradient = objective(proposed, X, weights)
            evaluations += 1
            if new_value <= value + 1e-4 * length * slope:
                break
            length *= 0.5
        else:
            reason = 'line_search_failed'
            break
        step, change = proposed - w, new_gradient - gradient
        curvature = float(np.dot(step, change))
        if curvature > 1e-12:
            memory.append((step, change, 1.0 / curvature))
            memory = memory[-10:]
        w, value, gradient = proposed, new_value, new_gradient
    return w, {'converged': reason == 'gradient_tolerance', 'stop_reason': reason,
        'iterations': len(trace) - 1, 'final_loss': value,
        'gradient_inf': float(np.abs(gradient).max()), 'evaluations': evaluations,
        'maxiter': maxiter, 'gtol': gtol, 'lambda': LAMBDA, 'zero_initialization': True,
        'ordered_pairs_implicitly_averaged': True, 'trace': trace}


def model_scores(bank, x, parsed, groups, grouped, model):
    n, dim = len(bank), len(model['weights'])
    scores = np.full((n, 4), -1e100, dtype=np.float64)
    for q in range(n):
        if not groups[q]:
            scores[q] = 0
            continue
        if model['mode'] == 'candidate_pair':
            for j, answer in enumerate(parsed[q]):
                if answer is not None:
                    scores[q, j] = ((x[q, j] - model['mean']) / model['scale']) @ model['weights']
        else:
            for g, (_, members) in enumerate(groups[q]):
                value = ((grouped[q][g][:dim] - model['mean']) / model['scale']) @ model['weights']
                scores[q, members] = value
    return scores


def choose_scores(bank, x, parsed, scores):
    selected = []
    for q, row in enumerate(bank):
        if not any(a is not None for a in parsed[q]):
            selected.append(0)
            continue
        best = [j for j in range(4) if parsed[q][j] is not None and scores[q, j] == scores[q].max()]
        selected.append(int(min(best, key=lambda j: (
            parsed[q][j], row['candidates'][j]['text'], x[q, j, :1024].tobytes()))))
    return selected


def load_model(checkpoint):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    mode = config['mode']
    if mode not in MODES or config['method'] != 'v6_answer_group_pair':
        raise ValueError('Unsupported group-model checkpoint')
    dim = 1032 if mode == 'group_mean_context' else WIDTH
    with np.load(checkpoint / 'model.npz', allow_pickle=False) as z:
        model = {k: z[k].copy() for k in ('weights', 'mean', 'scale')}
    if any(v.shape != (dim,) or not np.isfinite(v).all() for v in model.values()) or not (model['scale'] > 0).all():
        raise ValueError('Invalid model arrays')
    return {**model, 'mode': mode}


def predict(bank, arrays, checkpoint):
    model = load_model(checkpoint)
    x, parsed, groups = build_inputs(bank, arrays)
    grouped = group_values(x, groups)
    scores = model_scores(bank, x, parsed, groups, grouped, model)
    return choose_scores(bank, x, parsed, scores)
