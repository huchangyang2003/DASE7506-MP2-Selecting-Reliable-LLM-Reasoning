"""Input-only candidate features; no reference, ID, split or file access."""

from collections import Counter
from fractions import Fraction
import re

import numpy as np

from common import parse_answer

NUMBER = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')
WORD = re.compile(r'[a-z]+')
BASIC_NAMES = ['valid_answer', 'answer_negative', 'answer_integer', 'answer_log_abs',
               'answer_in_question', 'question_number_count', 'number_reuse_ratio',
               'word_jaccard']
STYLE_NAMES = ['log_characters', 'log_words', 'log_lines', 'log_number_count',
               'boxed_marker', 'hash_marker', 'bold_count', 'equation_count',
               'fraction_marker_count', 'dollar_count', 'confidence_words', 'uncertainty_words']
GROUP_NAMES = ['answer_support', 'support_margin', 'valid_fraction', 'distinct_fraction',
               'mean_other_cosine', 'max_other_cosine', 'same_answer_cosine',
               'different_answer_cosine', 'support_times_distinct', 'support_times_cosine']


def numbers(text):
    result = []
    for token in NUMBER.findall(text):
        try:
            result.append(str(Fraction(token.replace(',', ''))))
        except (ValueError, ZeroDivisionError):
            pass
    return result


def build_features(bank, arrays, variant='group'):
    if variant not in ('individual', 'group', 'group_no_style'):
        raise ValueError('Unknown feature variant')
    embeddings = np.asarray(arrays['embeddings'], dtype=np.float64)
    if embeddings.ndim != 3 or embeddings.shape[:2] != (len(bank), 4):
        raise ValueError('Feature input alignment mismatch')
    with_style = variant != 'group_no_style'
    with_group = variant != 'individual'
    names = [f'embedding_{i}' for i in range(embeddings.shape[2])] + BASIC_NAMES
    if with_style:
        names += STYLE_NAMES
    if with_group:
        names += GROUP_NAMES
    result = np.empty((len(bank), 4, len(names)), dtype=np.float64)
    result[:, :, :embeddings.shape[2]] = embeddings
    norms = np.linalg.norm(embeddings, axis=2, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    similarities = normalized @ normalized.swapaxes(1, 2)
    for row_index, row in enumerate(bank):
        question = row['question'].lower()
        question_numbers = set(numbers(question))
        question_words = set(WORD.findall(question))
        texts = [candidate['text'] for candidate in row['candidates']]
        answers = [parse_answer(text) for text in texts]
        counts = Counter(answer for answer in answers if answer is not None)
        valid_count = sum(answer is not None for answer in answers)
        for index, (text, answer) in enumerate(zip(texts, answers)):
            lower = text.lower()
            numeric_tokens = numbers(text)
            word_set = set(WORD.findall(lower))
            numeric_answer = Fraction(answer) if answer is not None else Fraction(0)
            numeric_float = float(np.clip(float(numeric_answer), -1e100, 1e100))
            values = [float(answer is not None), float(numeric_answer < 0),
                      float(answer is not None and numeric_answer.denominator == 1),
                      np.log1p(abs(numeric_float)), float(answer in question_numbers),
                      float(len(question_numbers)),
                      sum(number in question_numbers for number in numeric_tokens) / max(1, len(numeric_tokens)),
                      len(word_set & question_words) / max(1, len(word_set | question_words))]
            if with_style:
                values += [np.log1p(len(text)), np.log1p(len(text.split())),
                           np.log1p(len(text.splitlines())), np.log1p(len(numeric_tokens)),
                           float('\\boxed' in text), float('####' in text), text.count('**'),
                           text.count('='), text.count('\\frac'), text.count('$'),
                           sum(lower.count(word) for word in ('clearly', 'therefore', 'thus', 'certainly')),
                           sum(lower.count(word) for word in ('maybe', 'perhaps', 'unsure', 'however'))]
            if with_group:
                support = counts.get(answer, 0) if answer is not None else 0
                rival = max((count for other, count in counts.items() if other != answer), default=0)
                others = [j for j in range(4) if j != index]
                same = [j for j in others if answer is not None and answers[j] == answer]
                different = [j for j in others if answers[j] is not None and answers[j] != answer]
                cosine = similarities[row_index, index]
                mean_cosine = float(np.mean(cosine[others]))
                values += [support / 4, (support - rival) / 4, valid_count / 4,
                           len(counts) / 4, mean_cosine, float(np.max(cosine[others])),
                           float(np.mean(cosine[same])) if same else 0.0,
                           float(np.mean(cosine[different])) if different else 0.0,
                           support * len(counts) / 16, support * mean_cosine / 4]
            result[row_index, index, embeddings.shape[2]:] = values
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite engineered features')
    return result, names
