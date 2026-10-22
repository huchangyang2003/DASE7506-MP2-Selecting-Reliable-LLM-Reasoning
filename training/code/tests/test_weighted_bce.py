"""Validate difficulty definition, loss scale and weighted backpropagation."""

import unittest

import numpy as np

from objectives import network_loss_and_gradients
from weighted_bce import disagreement_mask, question_weights, weighted_network_loss_and_gradients


class WeightedBCETests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(51)
        self.features = rng.normal(size=(3, 4, 3)) * 0.1
        self.labels = np.array([[1, 0, 0, 0], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=float)
        self.params = {'W1': rng.normal(size=(3, 2)) * 0.1, 'b1': np.array([0.8, 1.0]),
                       'W2': np.array([0.2, -0.3]), 'b2': np.array(0.1)}

    def test_equivalent_answers_and_invalids_do_not_inflate_difficulty(self):
        def row(texts):
            return {'candidates': [{'text': text} for text in texts]}
        bank = [row(['#### 1/2', '#### 0.5', 'missing', 'missing']),
                row(['#### 1/2', '#### 2', 'missing', '#### 0.5'])]
        np.testing.assert_array_equal(disagreement_mask(bank), [False, True])

    def test_weight_normalization_preserves_relative_ratios(self):
        hard = np.array([True, False, False, True, False])
        weights, _ = question_weights(hard, 'hard_x4')
        self.assertAlmostEqual(float(weights.mean()), 1, places=6)
        self.assertAlmostEqual(float(weights[0] / weights[1]), 4)
        only, _ = question_weights(hard, 'hard_only')
        np.testing.assert_array_equal(only[~hard], 0)
        self.assertAlmostEqual(float(only.mean()), 1)

    def test_uniform_is_exact_original_bce(self):
        original = network_loss_and_gradients(self.params, self.features, self.labels, 'bce')
        weighted = weighted_network_loss_and_gradients(self.params, self.features, self.labels, np.ones(3))
        self.assertEqual(original[0], weighted[0])
        for name in self.params:
            np.testing.assert_array_equal(original[1][name], weighted[1][name])

    def test_weighted_network_gradient_by_finite_difference(self):
        weights = np.array([2.0, 0.0, 1.0])
        _, gradients, _ = weighted_network_loss_and_gradients(self.params, self.features, self.labels, weights)
        for name in self.params:
            numeric = np.zeros_like(self.params[name])
            for index in np.ndindex(numeric.shape):
                self.params[name][index] += 1e-6
                upper = weighted_network_loss_and_gradients(self.params, self.features, self.labels, weights)[0]
                self.params[name][index] -= 2e-6
                lower = weighted_network_loss_and_gradients(self.params, self.features, self.labels, weights)[0]
                self.params[name][index] += 1e-6
                numeric[index] = (upper - lower) / 2e-6
            np.testing.assert_allclose(gradients[name], numeric, rtol=1e-4, atol=1e-8)

    def test_zero_weights_remove_only_data_loss(self):
        total, gradients, data = weighted_network_loss_and_gradients(
            self.params, self.features, self.labels, np.zeros(3), l2=0.001)
        self.assertEqual(data, 0)
        self.assertGreater(total, 0)
        np.testing.assert_allclose(gradients['W1'], 0.001 * self.params['W1'])
        np.testing.assert_array_equal(gradients['b1'], np.zeros(2))
        self.assertEqual(float(gradients['b2']), 0)


if __name__ == '__main__':
    unittest.main()
