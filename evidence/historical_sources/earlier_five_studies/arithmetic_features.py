"""Bounded numeric-equality checks. Evidence only, never a hard answer filter."""
import ast
from fractions import Fraction
import re
import numpy as np
from common import parse_answer

COVERAGE_NAMES = ['log_equals', 'log_checked', 'checked_fraction', 'log_skipped', 'no_checked', 'truncated']
CONSISTENCY_NAMES = ['log_exact', 'log_rounding_only', 'log_bad', 'exact_fraction', 'accepted_fraction',
                     'bad_fraction', 'max_relative_error', 'mean_relative_error',
                     'has_final_and_checked', 'last_rhs_matches_final', 'rhs_matches_final_fraction']
MAX_TEXT = 20000
MAX_EQUATIONS = 128
MAX_EXPRESSION = 256
MAX_VALUE = Fraction(10**12)
MAX_DENOMINATOR = 10**12
NUMERIC_SPAN = re.compile(r'[0-9.,()+*/^% \t\-]+')


def evaluate_numeric(expression):
    """No evaluation of Python code: parse, whitelist and evaluate bounded AST."""
    if not expression or len(expression) > MAX_EXPRESSION:
        raise ValueError('Expression length')
    expression = expression.strip().replace('^', '**')
    expression = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', expression)
    if not re.fullmatch(r'[0-9.()+*/ \t\-]+', expression):
        raise ValueError('Non-numeric expression')
    tree = ast.parse(expression, mode='eval')
    if len(list(ast.walk(tree))) > 100:
        raise ValueError('AST too large')

    def bounded(v):
        if abs(v) > MAX_VALUE or v.denominator > MAX_DENOMINATOR:
            raise ValueError('Numeric bound')
        return v

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            token = ast.get_source_segment(expression, node)
            if token is None or len(token) > 32:
                raise ValueError('Literal bound')
            return bounded(Fraction(token))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = visit(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): v = a + b
            elif isinstance(node.op, ast.Sub): v = a - b
            elif isinstance(node.op, ast.Mult): v = a * b
            elif isinstance(node.op, ast.Div): v = a / b
            elif isinstance(node.op, ast.Pow):
                if b.denominator != 1 or abs(b) > 6 or abs(a) > 10**6:
                    raise ValueError('Exponent bound')
                v = a ** int(b)
            else: raise ValueError('Unsupported operator')
            return bounded(v)
        raise ValueError('Unsupported AST')
    return visit(tree)


def normalize(text):
    text = text[:MAX_TEXT]
    text = re.sub(r'\*\*([^*\n]+)\*\*', r'\1', text)
    text = re.sub(r'(?m)^\s*[-*•]\s+', '', text)
    text = re.sub(r'(?m)^\s*\d+[.)]\s+', '', text)
    text = re.sub(r'%\s+of\s+', '% * ', text, flags=re.IGNORECASE)
    for old, new in [('×', '*'), ('÷', '/'), ('−', '-'), ('–', '-'),
                     ('\\times', '*'), ('\\cdot', '*'), ('\\div', '/'),
                     ('\\left', ''), ('\\right', ''), ('\\%', '%'),
                     ('\\,', ' '), ('\\!', ' '), ('\\;', ' '), ('\\:', ' ')]:
        text = text.replace(old, new)
    for _ in range(8):
        old = text
        text = re.sub(r'\\(?:dfrac|tfrac|frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'((\1)/(\2))', text)
        text = re.sub(r'\\boxed\s*\{([^{}]*)\}', r'(\1)', text)
        if text == old: break
    # Unit/label commands are boundaries, not silently removed from expressions.
    text = re.sub(r'\\(?:text|mathrm|operatorname)\s*\{[^{}]*\}', '|', text)
    text = re.sub(r'(?<![\d.])\d{1,3}(?:,\d{3})+(?![\d.])', lambda m: m[0].replace(',', ''), text)
    return text.replace('$', ' ').replace('\\[', '\n').replace('\\]', '\n')


def extract_side(segment, side):
    if side == 'left':
        match = re.search(r'[0-9.,()+*/^% \t\-]+$', segment)
    else:
        match = NUMERIC_SPAN.match(segment)
    if not match:
        raise ValueError('No numeric side')
    expression = match[0].strip()
    if not expression:
        raise ValueError('Empty side')
    prefix, suffix = segment[:match.start()], segment[match.end():]
    if side == 'left':
        if re.search(r'\d|\b(?:of|half|twice|double|triple)\b', prefix, re.IGNORECASE):
            raise ValueError('Unparsed numeric or word-arithmetic prefix')
        if prefix and prefix[-1].isalnum() and not match[0].startswith((' ', '\t')):
            raise ValueError('Adjacent variable/word')
        # Avoid turning x + 20 into the constant +20, or x - 20 into -20.
        if expression.startswith(('+', '-')) and re.search(r'[A-Za-z0-9]\s*$', prefix):
            raise ValueError('Variable expression fragment')
        if '\\' in prefix[-1:]:
            raise ValueError('Unsupported command')
    elif suffix:
        if not match[0].endswith((' ', '\t')) and suffix[0].isalnum():
            raise ValueError('Adjacent variable/word')
        if re.search(r'[0-9=+*/^%]', suffix):
            raise ValueError('Unparsed expression tail')
        if re.match(r'\s*[A-Za-z]\s*$', suffix):
            raise ValueError('Single variable tail')
        if suffix.startswith('\\'):
            raise ValueError('Unsupported command')
    return expression, evaluate_numeric(expression)


def inspect_candidate(text):
    normalized = normalize(str(text))
    total = normalized.count('=')
    equations = []
    positions = [m.start() for m in re.finditer('=', normalized)][:MAX_EQUATIONS]
    for p in positions:
        # Numeric formula is local to a line and adjacent equals signs.
        start = max(normalized.rfind('\n', 0, p), normalized.rfind('=', 0, p)) + 1
        ends = [x for x in (normalized.find('\n', p + 1), normalized.find('=', p + 1)) if x >= 0]
        end = min(ends) if ends else len(normalized)
        try:
            left, a = extract_side(normalized[start:p], 'left')
            right, b = extract_side(normalized[p + 1:end], 'right')
            # 8/25*100=32% commonly uses % as an answer unit. Its right-hand
            # scalar can mean either 32 or .32; skip instead of declaring an error.
            if re.fullmatch(r'[-+]?\d+(?:\.\d+)?\s*%', right):
                continue
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
            continue
        difference = abs(a - b)
        tolerance = Fraction(1, 200) + max(abs(a), abs(b)) / 10**6
        relative = float(difference / max(Fraction(1), abs(a), abs(b)))
        equations.append({'left': left, 'right': right, 'lhs_value': str(a), 'rhs_value': str(b),
                          'exact': a == b, 'accepted': difference <= tolerance,
                          'relative_error': min(relative, 1.0)})
    checked = len(equations)
    exact = sum(e['exact'] for e in equations)
    accepted = sum(e['accepted'] for e in equations)
    parsed = parse_answer(str(text))
    final = Fraction(parsed) if parsed is not None else None
    matches = [final is not None and Fraction(e['rhs_value']) == final for e in equations]
    coverage = [np.log1p(total), np.log1p(checked), checked / max(total, 1), np.log1p(total - checked),
                float(checked == 0), float(len(str(text)) > MAX_TEXT or total > MAX_EQUATIONS)]
    residuals = [e['relative_error'] for e in equations]
    consistency = [np.log1p(exact), np.log1p(accepted - exact), np.log1p(checked - accepted),
                   exact / max(checked, 1), accepted / max(checked, 1), (checked - accepted) / max(checked, 1),
                   max(residuals, default=0), float(np.mean(residuals)) if residuals else 0,
                   float(final is not None and checked > 0), float(matches[-1]) if matches else 0,
                   sum(matches) / max(checked, 1)]
    return {'equals': total, 'checked': checked, 'exact': exact, 'accepted': accepted,
            'bad': checked - accepted, 'parsed_final': parsed, 'equations': equations,
            'coverage': coverage, 'consistency': consistency}


def build_arithmetic_features(bank, mode='arithmetic', details=False):
    if mode not in ('coverage', 'arithmetic'):
        raise ValueError('Unknown arithmetic mode')
    names = COVERAGE_NAMES + (CONSISTENCY_NAMES if mode == 'arithmetic' else [])
    result = np.empty((len(bank), 4, len(names)), dtype=float)
    records = []
    for q, row in enumerate(bank):
        group = []
        for j, candidate in enumerate(row['candidates']):
            record = inspect_candidate(candidate['text'])
            result[q, j] = record['coverage'] + (record['consistency'] if mode == 'arithmetic' else [])
            if details: group.append(record)
        if details: records.append(group)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite arithmetic features')
    return (result, names, records) if details else (result, names)
