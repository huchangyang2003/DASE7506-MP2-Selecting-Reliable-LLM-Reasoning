"""Frozen feature extraction shared by training and inference (NumPy only).

Two families of evidence are produced for every candidate:

* ``text``      -- cheap surface signals from the written solution, including a
  programmatic check of every arithmetic step it states, and how much its final
  answer agrees with the other three candidates (self-consistency).
* ``embedding`` -- the supplied 1024-d representation, plus its deviation from the
  group mean, which strips away the shared question component.

Both are computed without reference answers, so they are usable at evaluation time.
"""
import itertools
import re
from fractions import Fraction as _Fraction

import numpy as np

# ---------------------------------------------------------------- answer parsing
_NUM = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')


def parse_answer(text):
    """Mirror of common.parse_answer (kept local so inference is self-contained)."""
    from fractions import Fraction
    text = str(text)
    tail = text.rsplit('####', 1)[1] if '####' in text else None
    if tail is None:
        boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
        tail = boxed[-1] if boxed else ''
    match = re.search(_NUM, tail)
    if not match:
        return None
    try:
        return str(Fraction(match.group().replace(',', '')))
    except (ValueError, ZeroDivisionError):
        return None


def answer_key(text):
    """Coarse grouping key for answers: numeric value bucket, or the raw string."""
    from fractions import Fraction
    a = parse_answer(text)
    if a is None:
        return None
    try:
        return float(Fraction(a))
    except (ValueError, ZeroDivisionError):
        return None


# ------------------------------------------------------- arithmetic verification
_FRACTION = re.compile(r'\\[a-z]*frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}')
_LATEX_NOISE = [
    (r'\\text\{[^}]*\}', ' '), (r'\\mathrm\{[^}]*\}', ' '), (r'\\cdot', '*'),
    (r'\\times', '*'), (r'\\div', '/'), (r'\\%', ''), (r'\\!', ''), (r'\\,', ' '),
    (r'\\;', ' '), (r'\\quad', ' '), (r'\\qquad', ' '), (r'\$', ' '),
    (r'\\left', ' '), (r'\\right', ' '), (r'\\boxed', ' '),
    (r'\\overline\{[^}]*\}', ' '),
    (r'\{', '('), (r'\}', ')'), (r'&', ' '), (r'\\', ' '),
]


def _fractions_to_parens(text):
    """\\frac{40}{0.25} -> ((40)/(0.25)) so the division stays checkable."""
    previous = None
    for _ in range(4):
        if previous == text:
            break
        previous = text
        text = _FRACTION.sub(lambda m: f'(({m.group(1)})/({m.group(2)}))', text)
    return text


# A maximal run of expression tokens, e.g. "((40)/(0.25))" or "10 + 15 + 5".
# Crucially this skips stray punctuation such as the '.' at the end of a sentence,
# which would otherwise glue unrelated fragments together.
_RUN = re.compile(r'(?:\d+(?:\.\d+)?|[-+*/()])(?:[ \t]*(?:\d+(?:\.\d+)?|[-+*/()]))*')
_ALPHABET = re.compile(r'[A-Za-z]+')
# Require the left side to contain an operator before treating a line as a step.
REQUIRE_COMPOUND_LHS = True


def _clean(text):
    """Reduce a solution to lines of checkable arithmetic.

    Words are dropped line by line, so unit labels ("15 dollars per hour * 8 hours
    = 120") do not truncate the expression, while prose without an '=' sign cannot
    be mistaken for a calculation.
    """
    from fractions import Fraction
    out = text.replace('−', '-').replace('–', '-').replace('—', '-')
    out = out.replace('×', '*').replace('÷', '/').replace('·', '*')
    # thousands separators only: a lone comma (list separator) must survive so it
    # can still separate two unrelated numbers
    out = re.sub(r'(?<=\d),(?=\d{3}\b)', '', out)
    out = _fractions_to_parens(out)
    out = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'((\1)/100)', out)
    out = re.sub(r'\)\s*of\s*(\d+(?:\.\d+)?)', r')*(\1)', out)
    for pattern, rep in _LATEX_NOISE:
        out = re.sub(pattern, rep, out)
    out = out.replace('%', ' ').replace('$', ' ').replace('#', ' ')
    lines = []
    for line in out.splitlines():
        if '=' not in line:
            continue
        lines.append(_ALPHABET.sub(' ', line))
    return '\n'.join(lines)


# ------------------------------------------------------- expression evaluation
_NUM_TOKEN = re.compile(r'\d+(?:\.\d+)?')


def _tokenize(text):
    tokens = []
    pos = 0
    while pos < len(text):
        ch = text[pos]
        if ch.isspace():
            pos += 1
            continue
        if ch.isdigit():
            m = _NUM_TOKEN.match(text, pos)
            tokens.append(('n', _Fraction(m.group())))
            pos = m.end()
            continue
        if ch in '+-*/()':
            tokens.append(('o', ch))
            pos += 1
            continue
        raise ValueError(f'unexpected character {ch!r}')
    return tokens


class _Parser:
    """Recursive descent: expr -> term (('+'|'-') term)*, term -> factor (('*'|'/')
    factor)*, factor -> '-' atom, atom -> NUMBER | '(' expr ')'."""

    def __init__(self, tokens):
        self.t = tokens
        self.i = 0

    def peek(self):
        return self.t[self.i][1] if self.i < len(self.t) else None

    def take(self):
        token = self.t[self.i]
        self.i += 1
        return token

    def expr(self):
        value = self.term()
        while self.peek() in ('+', '-'):
            op = self.take()[1]
            value = value + self.term() if op == '+' else value - self.term()
        return value

    def term(self):
        """factor (('*'|'/') factor | implicit '(' multiplication)*."""
        value = self.factor()
        while True:
            following = self.peek()
            if following in ('*', '/'):
                op = self.take()[1]
                rhs = self.factor()
                if op == '*':
                    value = value * rhs
                else:
                    if rhs == 0:
                        raise ZeroDivisionError('divide by zero')
                    value = value / rhs
            elif following == '(':
                # implicit multiplication, as in the common "3(18) - 5"
                self.take()
                inner = self.expr()
                if self.peek() != ')':
                    raise ValueError('unbalanced parentheses')
                self.take()
                value = value * inner
            else:
                return value

    def factor(self):
        if self.peek() in ('-', '+'):
            sign = -1 if self.take()[1] == '-' else 1
            return sign * self.factor()
        return self.atom()

    def atom(self):
        if self.i >= len(self.t):
            raise ValueError('empty expression')
        kind, value = self.take()
        if value == '(':
            inner = self.expr()
            if self.peek() != ')':
                raise ValueError('unbalanced parentheses')
            self.take()
            return inner
        if kind != 'n':
            # an operator where a value was expected: not a real expression
            raise ValueError(f'unexpected {value!r}')
        return value

    def parse(self):
        value = self.expr()
        if self.i != len(self.t):
            raise ValueError('trailing tokens')
        return value


def _is_operator(token):
    return token[0] == 'o' and token[1] in '+-*/'


def _juxtaposed(left, right):
    """True when two tokens sit next to each other with no operator between them
    (e.g. "8 8 + 2"), which means the run swallowed more than one expression."""
    l_value, r_value = left[1], right[1]
    # "3(18)" is multiplication and is handled by the parser, but two bare numbers
    # side by side means the run swallowed two separate expressions
    return (left[0] == 'n' and right[0] == 'n') or (l_value == ')' and right[0] == 'n')


def _trim_run(tokens, from_left=True):
    """Keep the longest infix-valid slice anchored at the requested end.

    Scanning backwards must still test pairs in their original order, otherwise a
    harmless "number )" pair would look like a violation when reversed.
    """
    kept = []
    if from_left:
        for token in tokens:
            if kept and _juxtaposed(kept[-1], token):
                break
            kept.append(token)
        return kept
    for token in reversed(tokens):
        if kept and _juxtaposed(token, kept[-1]):
            break
        kept.append(token)
    return kept[::-1]


def _value_of(tokens):
    try:
        return _Parser(tokens).parse()
    except (ValueError, ZeroDivisionError, IndexError, TypeError, AttributeError):
        return None


def _evaluate(text):
    """Exact rational value of an arithmetic string, or None if unparseable."""
    try:
        return _value_of(_tokenize(text))
    except ValueError:
        return None


def scan_equations(text):
    """Yield (lhs_value, rhs_value) for every checkable '<expr> = <expr>' step.

    Only self-contained steps are accepted: both sides must parse as complete
    arithmetic expressions, otherwise nothing is recorded. That keeps the verifier
    from accusing a solution of bad arithmetic merely because our parser could not
    read it -- a false accusation would be worse than a missed step.
    """
    clean = _clean(text)
    by_end, by_start, bounds = {}, {}, {}
    for m in _RUN.finditer(clean):
        tokens = _tokenize(m.group())
        by_end[m.end()] = tokens
        by_start[m.start()] = tokens
        bounds[m.start()] = m.end()
        bounds[m.end()] = m.start()
    out = []
    size = len(clean)
    for m in re.finditer(r'(?<!=)=(?!=)', clean):
        left, right = m.start(), m.end()
        while left > 0 and clean[left - 1] in ' \t':     # skip padding around '='
            left -= 1
        while right < size and clean[right] in ' \t':
            right += 1
        left_tokens = by_end.get(left)
        right_tokens = by_start.get(right)
        if not left_tokens or not right_tokens:
            continue
        # refuse if the sides touch something that suggests the run was cut out of
        # a longer expression: a superscript, or a stray word/number
        if _neighbour(clean, bounds[left], -1) or _neighbour(clean, bounds[right], 1):
            continue
        lhs_tokens = _trim_run(left_tokens, from_left=False)
        rhs_tokens = _trim_run(right_tokens, from_left=True)
        # neither side may start or end on a dangling operator: that signals the
        # run is only a fragment of the real expression
        if not lhs_tokens or not rhs_tokens:
            continue
        if _is_operator(lhs_tokens[0]) or _is_operator(rhs_tokens[-1]):
            continue
        if REQUIRE_COMPOUND_LHS and len(lhs_tokens) < 3:
            # a bare number on the left is usually an item label ("3 dozen = 3*12"),
            # not a step being claimed as arithmetic
            continue
        lhs, rhs = _value_of(lhs_tokens), _value_of(rhs_tokens)
        if lhs is None or rhs is None:
            continue
        out.append((lhs, rhs))
    return out


def _neighbour(clean, index, direction):
    """True when the non-blank character adjacent to a run boundary casts doubt on
    the run being a complete expression (word, digit, superscript, power)."""
    position = index - 1 if direction < 0 else index
    size = len(clean)
    while 0 <= position < size and clean[position] in ' \t\n':
        position += direction
    if not 0 <= position < size:
        return False
    ch = clean[position]
    return ch.isalnum() or ch in '^_'


def check_arithmetic(text):
    """Return (n_equations, n_valid, n_invalid) for stated arithmetic steps.

    Every explicit "a op b ... = c" statement in the solution is recomputed with
    exact rational arithmetic, so a wrong intermediate step is detectable without
    any language model. Returns the counts so callers can form rates.
    """
    total = valid = 0
    for lhs, rhs in scan_equations(text):
        total += 1
        valid += int(abs(lhs - rhs) < _Fraction(1, 1000))
    return total, valid, total - valid


# --------------------------------------------------- answer-trace consistency
def check_trace_support(text):
    """Is the final answer actually *reached* by the written calculations?

    Returns [last_step_matches, n_derivations, reaches_answer, dangling_answers]:
      last_step_matches -- the final stated result equals the final answer
      n_derivations     -- how many stated results equal the final answer
      reaches_answer    -- any calculation produces the answer (>=1 derivation)
      dangling_answers  -- results that are stated but never used later
    """
    results = [rhs for _, rhs in scan_equations(text)]
    answer = parse_answer(text)
    if answer is None or '/' in str(answer):
        return [0.0, 0.0, 0.0, 0.0]
    target = _Fraction(answer)
    matches = [r for r in results if abs(r - target) < _Fraction(1, 1000)]
    return [
        float(bool(results) and abs(results[-1] - target) < _Fraction(1, 1000)),
        float(len(matches)),
        float(len(matches) > 0),
        float(sum(1 for r in results if abs(r - target) >= _Fraction(1, 1000))),
    ]


# --------------------------------------------------------------- surface features
_SURFACE_NAMES = [
    'chars', 'words', 'lines', 'bullets', 'n_numbers', 'n_distinct_numbers',
    'n_op', 'n_eq_total', 'n_eq_valid', 'n_eq_invalid', 'eq_valid_frac',
    'has_boxed', 'has_hash', 'has_display_math', 'n_step_words', 'has_final_word',
    'has_units_word', 'frac_vs_int_answer', 'answer_is_integer', 'answer_negative',
    # does the written trace actually reach the answer it claims?
    'trace_last_match', 'trace_n_match', 'trace_reaches', 'trace_dangling',
    # question grounding: did the solution actually consume the given numbers?
    'question_nums_used_frac', 'question_nums_unused', 'answer_equals_question_num',
    'answer_lt_max_question_num', 'question_num_mentions',
    # how much of the solution is prose vs calculation
    'digit_density', 'newline_density', 'last_line_len',
    # embedding geometry inside the group
    'sim_to_centroid', 'sim_to_nearest', 'sim_max_minus_mean',
]


def _question_numbers(question):
    return [float(x) for x in re.findall(r'\d+(?:\.\d+)?', question.replace(',', ''))]


def _surface_features(text, question_numbers, unit_row, index):
    words = text.split()
    lower = text.lower()
    body_numbers = re.findall(r'\d+(?:\.\d+)?', text.replace(',', ''))
    total, valid, invalid = check_arithmetic(text)
    answer = parse_answer(text)
    is_frac = answer is not None and '/' in answer
    try:
        answer_float = float(answer) if answer is not None and '/' not in str(answer) else 0.0
    except ValueError:
        answer_float = 0.0
    body_set = set(body_numbers)
    trace = check_trace_support(text)
    if question_numbers:
        used = sum(1 for v in question_numbers if v in body_set)
        used_frac = used / len(question_numbers)
        unused = len(question_numbers) - used
        mentions = sum(body_numbers.count(v) for v in question_numbers)
        ans_in_q = float(answer_float != 0.0
                         and any(abs(answer_float - v) < 1e-9 for v in question_numbers))
        lt_max = float(answer_float <= max(question_numbers) + 1e-9)
    else:
        used_frac, unused, mentions, ans_in_q, lt_max = 0.0, 0.0, 0.0, 0.0, 0.0
    sims = unit_row @ unit_row[index]
    others = [j for j in range(len(sims)) if j != index]
    return [
        len(text),
        len(words),
        text.count('\n') + 1,
        sum(1 for ln in text.splitlines() if ln.strip().startswith(('-', '*'))),
        len(body_numbers),
        len(body_set),
        sum(text.count(op) for op in ('+', '-', '*', '/')),
        total,
        valid,
        invalid,
        valid / total if total else 0.0,
        float('\\boxed' in text),
        float('####' in text),
        float('$$' in text),
        sum(lower.count(w) for w in ('step', 'first', 'then', 'next', 'finally')),
        float(any(w in lower for w in ('answer', 'total', 'result'))),
        float(any(w in lower for w in ('miles', 'dollars', 'minutes', 'hours',
                                       'apples', 'kg', 'percent'))),
        float(is_frac),
        float(answer is not None and '.' not in str(answer) and '/' not in str(answer)),
        float(answer_float < 0) if answer is not None else 0.0,
        *trace,
        used_frac,
        unused,
        ans_in_q,
        lt_max,
        mentions,
        sum(ch.isdigit() for ch in text) / max(len(text), 1),
        text.count('\n') / max(len(words), 1),
        len(text.splitlines()[-1]) if text.strip() else 0,
        float(sims.mean()),
        float(max(sims[j] for j in others)) if others else 0.0,
        float(max(sims[j] for j in others) - np.mean([sims[j] for j in others])) if others else 0.0,
    ]


# ---------------------------------------------------------- consensus features
_CONSENSUS_NAMES = [
    'vote_count', 'vote_share', 'is_plurality', 'is_unique', 'group_distinct',
    'answer_rank_value', 'answer_rel_max', 'cluster_embed_sim',
    'sim_same_answer', 'sim_other_answer', 'sim_same_minus_other',
]


def _consensus_features(row_values, embedded_row):
    """row_values: per-candidate floats-or-None; embedded_row: [k,1024] array.

    The group size is read from the input rather than assumed, so the same frozen
    bundle also works on a bank with a different number of candidates per question.
    """
    n = len(row_values)
    valid = [(v is not None) for v in row_values]
    out = []
    unit = embedded_row / (np.linalg.norm(embedded_row, axis=1, keepdims=True) + 1e-9)
    sim = unit @ unit.T
    for i in range(n):
        if not valid[i]:
            out.append([0.0] * len(_CONSENSUS_NAMES))
            continue
        same = sum(1 for j in range(n)
                   if valid[j] and abs(row_values[j] - row_values[i]) < 1e-9)
        distinct = len({round(v, 9) for v, j in zip(row_values, range(n)) if valid[j]})
        values = [v for v in row_values if v is not None]
        rank = sorted(values).index(row_values[i]) / max(len(values) - 1, 1)
        rel = row_values[i] / max(abs(max(values, key=abs)), 1e-9)
        others = [j for j in range(n) if j != i]
        cluster = float(np.mean([sim[i, j] for j in others])) if others else 0.0
        same_peers = [j for j in others
                      if valid[j] and abs(row_values[j] - row_values[i]) < 1e-9]
        other_peers = [j for j in others
                       if valid[j] and abs(row_values[j] - row_values[i]) >= 1e-9]
        sim_same = float(np.mean([sim[i, j] for j in same_peers])) if same_peers else 0.0
        sim_other = float(np.mean([sim[i, j] for j in other_peers])) if other_peers else 0.0
        pluralities = [sum(1 for k in range(n) if valid[k]
                           and abs(row_values[k] - row_values[j]) < 1e-9)
                       for j in range(n) if valid[j]]
        out.append([same, same / n, float(same >= max(pluralities)),
            float(same == 1), distinct, rank, rel, cluster,
            sim_same, sim_other, sim_same - sim_other])
    return out


# ------------------------------------------------------------------- public API
TEXT_FEATURE_NAMES = _SURFACE_NAMES + _CONSENSUS_NAMES
# Numeric features that also get a within-group standardised twin: only relative
# comparisons matter when one candidate must be chosen out of four.
STANDARDISE_SLICE = slice(0, len(_SURFACE_NAMES))


def build_group_features(question, candidates, embeds):
    """Feature matrix for one comparison group of any size -> [k, F].

    Split out from `build_features` so training can also enumerate *subsets* of a
    group's candidates: each subset is an equally legitimate instance of the same
    decision problem, which is how scarce supervision gets multiplied (see
    `informative_subsets` and `train.py`).
    """
    unit = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-9)
    q_numbers = _question_numbers(question)
    values = [answer_key(c['text']) for c in candidates]
    consensus = _consensus_features(values, embeds)
    block = np.asarray([_surface_features(c['text'], q_numbers, unit, i) + cons
                        for i, (c, cons) in enumerate(zip(candidates, consensus))],
                       dtype=np.float64)
    core = block[:, STANDARDISE_SLICE]
    mu = core.mean(axis=0)
    sd = core.std(axis=0) + 1e-6
    return np.concatenate([block, (core - mu) / sd], axis=1)


def build_features(bank, arrays):
    """Return (X_text [N,4,F], X_embed [N,4,1024], X_dev [N,4,1024]).

    X_text concatenates surface + consensus features, followed by within-group
    standardised twins of the leading numeric surface statistics, so the selector
    can reason both absolutely ("this solution is long") and comparatively
    ("this solution is the longest of the four").
    """
    embedded = arrays['embeddings'].astype(np.float32)
    centred = embedded - embedded.mean(axis=1, keepdims=True)
    rows_text = [build_group_features(row['question'], row['candidates'], embedded[n])
                 for n, row in enumerate(bank)]
    return np.asarray(rows_text, dtype=np.float32), embedded, centred


# ------------------------------------------------------------------ augmentation
def informative_subsets(labels, sizes=(2, 3)):
    """Index tuples of the sub-groups inside one group that are worth training on.

    A four-candidate question already contains several smaller versions of the same
    decision problem -- the right answer against one distractor, against two, and so
    on. Each is a legitimate instance of "choose the good one", so enumerating them
    multiplies scarce supervision without introducing a single new question.

    A sub-group is kept only when it holds at least one correct and at least one
    incorrect candidate; otherwise there is no preference to learn from it.
    """
    n = len(labels)
    out = [tuple(range(n))]
    for size in sizes:
        if size >= n:
            continue
        for combo in itertools.combinations(range(n), size):
            sub = [labels[i] for i in combo]
            if any(sub) and not all(sub):
                out.append(combo)
    return out


def build_augmented_rows(bank, arrays, labels, rows, sizes=(2, 3), width=None):
    """Features for every informative sub-group of the given questions.

    Returns (X [M,W,d], y [M,W], seen [M,W]) with sub-groups padded to the widest
    group and flagged by ``seen``, so `model.fit` can ignore the padding. This is
    the training-side counterpart of `build_features`; inference never sees it.
    """
    embedded = arrays['embeddings'].astype(np.float32)
    if width is None:
        width = max(len(row['candidates']) for row in bank)
    blocks, targets, seen = [], [], []
    for row_index in rows:
        row = bank[row_index]
        candidates = row['candidates']
        emb = embedded[row_index]
        y_row = labels[row_index]
        for combo in informative_subsets(list(y_row), sizes):
            idx = list(combo)
            sub_emb = emb[idx]
            X_text = build_group_features(row['question'],
                                          [candidates[i] for i in idx], sub_emb)
            # deviating from the sub-group's own mean, not the original group's
            centred = sub_emb - sub_emb.mean(axis=0, keepdims=True)
            whole = np.concatenate([X_text, sub_emb, centred], axis=-1)
            block = np.zeros((width, whole.shape[-1]), dtype=np.float32)
            block[:len(idx)] = whole
            target = np.zeros(width, dtype=np.float64)
            target[:len(idx)] = y_row[idx]
            present = np.zeros(width, dtype=bool)
            present[:len(idx)] = True
            blocks.append(block)
            targets.append(target)
            seen.append(present)
    return np.stack(blocks), np.stack(targets), np.stack(seen)

