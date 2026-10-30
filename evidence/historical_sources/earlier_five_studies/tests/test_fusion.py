"""Check fold isolation, fold-only preprocessing and input-only fusion symmetry."""
import unittest
import numpy as np
from fusion_core import grouped_folds, fit_standardization, meta_features, answer_statistics, fixed_average
from stage3_logistic import fit_logistic, loss_gradient


class FusionTests(unittest.TestCase):
    def bank(self):
        return [{'question': f'q{i}', 'candidates': [{'text': t} for t in
                ['#### 0.5', '#### 1/2', '#### 2', 'invalid']]} for i in range(15)]

    def test_grouped_oof_no_question_overlap(self):
        bank = self.bank()
        bank += [{**bank[0], 'question': '  q0 \n'}]
        folds = grouped_folds(bank)
        self.assertEqual(folds[0], folds[-1])
        np.testing.assert_array_equal(folds, grouped_folds(bank))
        for fold in range(5):
            train = {' '.join(b['question'].split()) for b, f in zip(bank, folds) if f != fold}
            held = {' '.join(b['question'].split()) for b, f in zip(bank, folds) if f == fold}
            self.assertFalse(train & held)

    def test_preprocessing_ignores_heldout_rows(self):
        x = np.arange(48, dtype=float).reshape(3, 4, 4)
        mean, scale = fit_standardization(x, [0, 1])
        x[2] = 1e9
        other_mean, other_scale = fit_standardization(x, [0, 1])
        np.testing.assert_array_equal(mean, other_mean)
        np.testing.assert_array_equal(scale, other_scale)

    def test_equivalent_valid_answers_and_invalids(self):
        support, valid, same = answer_statistics(self.bank()[:1])
        np.testing.assert_array_equal(support, [[0.5, 0.5, 0.25, 0]])
        np.testing.assert_array_equal(valid, [[1, 1, 1, 0]])
        self.assertEqual(same[0, 3].sum(), 0)

    def test_fusion_features_permute_and_do_not_depend_on_id(self):
        bank = self.bank()[:2]
        a = np.array([[1, 3, 2, -4], [-1, 0, 2, 3]], dtype=float)
        b = -a * 0.7
        original = meta_features(bank, a, b)
        order = [2, 0, 3, 1]
        new = [{**row, 'id': 'never-seen', 'candidates': [row['candidates'][i] for i in order]} for row in bank]
        np.testing.assert_allclose(meta_features(new, a[:, order], b[:, order]), original[:, order], atol=1e-14)
        np.testing.assert_allclose(fixed_average(new, a[:, order], b[:, order]), fixed_average(bank, a, b)[:, order])

    def test_logistic_gradient_and_convergence(self):
        rng = np.random.default_rng(16)
        x = rng.normal(size=(40, 8))
        y = (x[:, 0] > 0).astype(float)
        theta = rng.normal(size=9) * 0.1
        _, gradient = loss_gradient(theta, x, y, 0.01)
        for i in range(9):
            delta = np.zeros(9); delta[i] = 1e-5
            numerical = (loss_gradient(theta + delta, x, y, 0.01)[0] -
                         loss_gradient(theta - delta, x, y, 0.01)[0]) / 2e-5
            self.assertAlmostEqual(numerical, gradient[i], places=7)
        fitted, result = fit_logistic(x, y, 0.01)
        self.assertTrue(result['converged'])
        self.assertGreater(np.mean((x @ fitted[:-1] + fitted[-1] > 0) == y), 0.9)


if __name__ == '__main__':
    unittest.main()
