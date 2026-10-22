"""Semantic edge cases for conservative, bounded arithmetic evidence."""
from fractions import Fraction
import unittest
import numpy as np
from arithmetic_features import evaluate_numeric, inspect_candidate, build_arithmetic_features


class ArithmeticTests(unittest.TestCase):
    def test_exact_decimal_fraction_and_precedence(self):
        self.assertEqual(evaluate_numeric('0.1+0.2'), Fraction(3, 10))
        self.assertEqual(evaluate_numeric('(2+3)*4/5'), Fraction(4))
        self.assertEqual(evaluate_numeric('25%*40'), Fraction(10))
        self.assertEqual(evaluate_numeric('2^3'), Fraction(8))

    def test_safe_rejection_and_bounds(self):
        for expression in ['__import__("os")', 'x+2', '2**1000000', '1/0', '9'*80, '2//1', 'abs(-1)']:
            with self.assertRaises((ValueError, SyntaxError, ZeroDivisionError)):
                evaluate_numeric(expression)

    def test_latex_chain_boxed_thousands_and_units(self):
        record = inspect_candidate(r'$\frac{10+15+5}{2} = \frac{30}{2} = 15$ miles.'+'\n#### 15')
        self.assertEqual(record['checked'], 2)
        self.assertEqual(record['exact'], 2)
        self.assertEqual(record['consistency'][-2], 1)
        other = inspect_candidate(r'1,000 \times 2 = \boxed{2000}'+'\n#### 2000')
        self.assertEqual(other['exact'], 1)

    def test_markdown_and_wrong_equation_soft_evidence(self):
        record = inspect_candidate('- **2 + 3 = 6**\n#### 6')
        self.assertEqual(record['bad'], 1)
        self.assertEqual(record['parsed_final'], '6')
        correct_final = inspect_candidate('2+3=6\n#### 5')
        self.assertEqual(correct_final['bad'], 1)
        self.assertEqual(correct_final['consistency'][-2], 0)

    def test_variables_implicit_multiplication_not_numeric_fragments(self):
        for text in ['x + 20 = 30', 'x-20=30', '2x + 3 = 7', '2 + x = 7',
                     '5 = 2x + 3', '5=2x', '5 = 2 x + 3', 'x**2 = 4', r'\sqrt{4} = 2']:
            self.assertEqual(inspect_candidate(text)['checked'], 0, text)

    def test_rounding_not_conflated_with_exact(self):
        record = inspect_candidate('1/3=0.33\n2+2=5')
        self.assertEqual((record['checked'], record['exact'], record['accepted'], record['bad']), (2, 0, 1, 1))

    def test_percent_of_and_word_arithmetic_boundaries(self):
        for text in ['2000 - 10% of 2000 = 1800', '60% of 2000 = 0.6*2000 = 1200']:
            record = inspect_candidate(text)
            self.assertGreater(record['checked'], 0)
            self.assertEqual(record['bad'], 0)
        for text in ['half of 28 = 14', '25 percent of 400 = 100', '10 plus 20 = 30']:
            self.assertEqual(inspect_candidate(text)['checked'], 0)

    def test_percent_answer_unit_is_ambiguous_not_an_error(self):
        for text in [r'8/25*100 = 32\%', '0.32 = 32%', '20*0.4=8%']:
            self.assertEqual(inspect_candidate(text)['checked'], 0)

    def test_zero_coverage_and_final_answer_is_not_a_gold_label(self):
        empty = inspect_candidate('There are no numeric equalities.\n#### 17')
        self.assertEqual(empty['checked'], 0)
        self.assertTrue(np.isfinite(empty['coverage'] + empty['consistency']).all())
        wrong_semantics = inspect_candidate('40/0.25=160\n#### 160')
        self.assertEqual(wrong_semantics['exact'], 1)
        self.assertEqual(wrong_semantics['consistency'][-2], 1)
        # This is only a candidate self-consistency signal, never proof it answers the question.

    def test_permutation_and_finite_features(self):
        candidates = [{'text': t} for t in ['2+3=5\n#### 5', '2+3=6\n#### 6', 'no formula', '1/3=0.33\n#### 0.33']]
        x, names = build_arithmetic_features([{'candidates': candidates}])
        order = [2, 0, 3, 1]
        y, other_names = build_arithmetic_features([{'id': 'unused', 'candidates': [candidates[i] for i in order]}])
        np.testing.assert_array_equal(y, x[:, order])
        self.assertEqual(names, other_names)
        self.assertTrue(np.isfinite(x).all())


if __name__ == '__main__':
    unittest.main()
