import io
import json
import unittest

import numpy as np

from algorithms.promptwise import promptwise, solve_lin_logistic_regression
from benchmarks.collect_qwen import numeric_answer
from benchmarks.retry_benchmark import request
from benchmarks.routers import DiscountedPromptWise


class RetryTests(unittest.TestCase):
    def router(self, discounted=False, discount=1):
        cls = DiscountedPromptWise if discounted else promptwise
        kwargs = dict(G=2, num_dim=2, rd_budget=3, model_cost=[1, 2], cost_para=.01,
                      reg_method='mle', tau_exp=1, exp_para=0)
        if discounted:
            kwargs['discount'] = discount
        return cls(**kwargs)

    def trained(self):
        r = self.router()
        for g, reward in [(0, 1), (1, 0)] * 8:
            r.update_stats(g, np.array([[1., 1.]]), reward)
            r.reset_rd_stats()
        return r

    def test_unit_discount_matches_original(self):
        a, b = self.router(), self.router(True)
        rng = np.random.default_rng(1)
        for t in range(20):
            x, g, y = rng.normal(size=(1, 2)), t % 2, int(t % 3 == 0)
            for r in (a, b):
                r.update_stats(g, x, y)
                r.reset_rd_stats()
            np.testing.assert_allclose(a.reg_model[g], b.reg_model[g], atol=1e-10)
            np.testing.assert_allclose(a.inv_Gram_mat[g], b.inv_Gram_mat[g], atol=1e-15)
        for r in (a, b):
            np.random.seed(123)
            selected = r.select_arm(np.ones((1, 2)))
            if r is a:
                expected = selected
            else:
                self.assertEqual(selected, expected)

    def test_discounted_gram_and_likelihood(self):
        r = self.router(True, .5)
        x = np.array([[1., 1.]])
        for y in [1, 1, 0]:
            r.update_stats(0, x, y)
            r.reset_rd_stats()
        expected = 1e6*np.eye(2) + 1.75*x.T@x
        np.testing.assert_allclose(np.linalg.inv(r.inv_Gram_mat[0]), expected)
        theta = solve_lin_logistic_regression(np.repeat(x, 3, axis=0), np.array([1, 1, 0]),
                                              sample_weight=np.array([.25, .5, 1.]))
        np.testing.assert_allclose(r.reg_model[0], theta)

    def test_attempt_and_cost_caps(self):
        x, y = np.ones((1, 2)), np.zeros((2, 5), dtype=int)
        r = self.trained()
        result = request(r, x, y, 'repeat_initial', 5, 5, 2.)
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(result['cost'], 2.)
        self.assertEqual(result['stop_reason'], 'cost_budget')
        result = request(self.trained(), x, y, 'repeat_initial', 5, 3, 100.)
        self.assertEqual(result['attempts'], 3)
        self.assertEqual(result['repeats'], 2)

    def test_success_stops_and_clears_request_state(self):
        r = self.trained()
        result = request(r, np.ones((1, 2)), np.ones((2, 5), dtype=int), 'original', 5, 3, 100.)
        self.assertEqual(result['attempts'], 1)
        self.assertEqual(result['success'], 1)
        self.assertFalse(r.rd_skip)
        self.assertEqual(r.rd_used_budget, 0)

    def test_switch_and_feedback_isolation(self):
        x = np.ones((1, 2))
        r = self.trained()
        before = r.visitation.copy()
        trace = io.StringIO()
        result = request(r, x, np.zeros((2, 5), dtype=int), 'switch_on_failure', 5, 5, 100., trace)
        self.assertEqual(json.loads(result['actions']), [0, 1])
        np.testing.assert_array_equal(r.visitation-before, [1, 1])
        self.assertEqual(result['stop_reason'], 'all_models_tried')
        self.assertEqual(len(trace.getvalue().splitlines()), 2)
        # Unchosen counterfactual rewards cannot change the first choice.
        for other_reward in [0, 1]:
            y = np.zeros((2, 5), dtype=int)
            y[1] = other_reward
            self.assertEqual(request(self.trained(), x, y, 'original', 5, 1, 100.)['first_arm'], 0)

    def test_strict_numeric_judge(self):
        self.assertEqual(numeric_answer('Work 17.\n#### 1,234.50'), numeric_answer('#### 1234.5'))
        self.assertIsNone(numeric_answer('There are 17 intermediate items.'))

    def test_paired_draws_across_attempt_caps(self):
        traces = []
        for cap in (1, 3):
            trace = io.StringIO()
            request(self.trained(), np.ones((1, 2)), np.zeros((2, 10), dtype=int),
                    'repeat_initial', 91, cap, 100., trace)
            traces.append([json.loads(line) for line in trace.getvalue().splitlines()])
        self.assertEqual(traces[0][0]['sample'], traces[1][0]['sample'])

    def test_unaffordable_selection_does_not_update(self):
        router = self.trained()
        before = router.visitation.copy()
        result = request(router, np.ones((1, 2)), np.zeros((2, 5), dtype=int),
                         'original', 5, 3, .5)
        self.assertEqual(result['attempts'], 0)
        self.assertEqual(result['cost'], 0)
        np.testing.assert_array_equal(before, router.visitation)

    def test_empirical_samples_not_reused_and_prefixes_paired(self):
        traces = []
        for cap in (3, 5):
            trace = io.StringIO()
            request(self.trained(), np.ones((1, 2)), np.zeros((2, 5), dtype=int),
                    'repeat_initial', 91, cap, 100., trace, sampling='without_replacement')
            samples = [json.loads(line)['sample'] for line in trace.getvalue().splitlines()]
            self.assertEqual(len(samples), len(set(samples)))
            traces.append(samples)
        self.assertEqual(traces[0], traces[1][:3])
        with self.assertRaises(ValueError):
            request(self.trained(), np.ones((1, 2)), np.zeros((2, 5), dtype=int),
                    'repeat_initial', 91, 6, 100., sampling='without_replacement')


if __name__ == '__main__':
    unittest.main()
