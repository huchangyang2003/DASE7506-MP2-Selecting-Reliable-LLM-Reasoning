"""Numerical and semantic checks for multi-positive ranking losses."""

import unittest

import numpy as np

from objectives import loss_and_gradient, network_loss_and_gradients

OBJECTIVES = ('bce', 'pairwise', 'listwise')


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1]], dtype=float)
        self.logits = np.random.default_rng(22).normal(size=(4, 4))

    def test_all_objective_gradients_by_finite_difference(self):
        for objective in OBJECTIVES:
            _, analytic = loss_and_gradient(self.logits, self.labels, objective)
            numeric = np.zeros_like(self.logits)
            for index in np.ndindex(self.logits.shape):
                delta = np.zeros_like(self.logits)
                delta[index] = 1e-6
                numeric[index] = (loss_and_gradient(self.logits + delta, self.labels, objective)[0]
                                  - loss_and_gradient(self.logits - delta, self.labels, objective)[0]) / 2e-6
            np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-8, err_msg=objective)

    def test_multi_positive_listwise_known_values(self):
        self.assertAlmostEqual(loss_and_gradient(np.zeros((1, 4)), self.labels[1:2], 'listwise')[0], np.log(2))
        self.assertAlmostEqual(loss_and_gradient(np.zeros((1, 4)), self.labels[:1], 'listwise')[0], np.log(4))

    def test_all_correct_questions_have_zero_ranking_loss_and_gradient(self):
        for objective in ('pairwise', 'listwise'):
            loss, gradient = loss_and_gradient(self.logits[-1:], self.labels[-1:], objective)
            self.assertEqual(loss, 0)
            np.testing.assert_array_equal(gradient, np.zeros((1, 4)))

    def test_ranking_is_invariant_to_per_question_score_shift(self):
        shifted = self.logits + np.array([2, -9, 4, 11])[:, None]
        for objective in ('pairwise', 'listwise'):
            old_loss, old_grad = loss_and_gradient(self.logits, self.labels, objective)
            new_loss, new_grad = loss_and_gradient(shifted, self.labels, objective)
            self.assertAlmostEqual(old_loss, new_loss)
            np.testing.assert_allclose(old_grad, new_grad, atol=1e-15)

    def test_candidate_permutation_follows_labels_and_gradient(self):
        order = [2, 0, 3, 1]
        for objective in OBJECTIVES:
            loss, gradient = loss_and_gradient(self.logits, self.labels, objective)
            shuffled_loss, shuffled_gradient = loss_and_gradient(self.logits[:, order], self.labels[:, order], objective)
            self.assertAlmostEqual(loss, shuffled_loss)
            np.testing.assert_allclose(shuffled_gradient, gradient[:, order], atol=1e-15)

    def test_extreme_logits_are_numerically_stable(self):
        logits = np.array([[1000., -1000., 999., -999.]])
        for objective in OBJECTIVES:
            loss, gradient = loss_and_gradient(logits, np.array([[0., 1., 0., 1.]]), objective)
            self.assertTrue(np.isfinite(loss))
            self.assertTrue(np.isfinite(gradient).all())

    def test_full_network_gradients_by_finite_difference(self):
        rng = np.random.default_rng(23)
        features = rng.normal(size=(2, 4, 3)) * 0.1
        labels = self.labels[:2]
        params = {'W1': rng.normal(size=(3, 2)) * 0.1, 'b1': np.array([0.8, 1.0]),
                  'W2': np.array([0.2, -0.3]), 'b2': np.array(0.1)}
        for objective in OBJECTIVES:
            _, analytic, _ = network_loss_and_gradients(params, features, labels, objective)
            for name in params:
                numeric = np.zeros_like(params[name])
                for index in np.ndindex(params[name].shape):
                    params[name][index] += 1e-6
                    upper = network_loss_and_gradients(params, features, labels, objective)[0]
                    params[name][index] -= 2e-6
                    lower = network_loss_and_gradients(params, features, labels, objective)[0]
                    params[name][index] += 1e-6
                    numeric[index] = (upper - lower) / 2e-6
                np.testing.assert_allclose(analytic[name], numeric, rtol=1e-4, atol=1e-8,
                                           err_msg=objective + ':' + name)


if __name__ == '__main__':
    unittest.main()
