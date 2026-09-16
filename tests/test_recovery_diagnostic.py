import copy
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from algorithms.promptwise import promptwise, solve_lin_logistic_regression
from benchmarks.recovery_diagnostic import DiscountAfterFailure, evaluate_case
from benchmarks.routers import scores
from benchmarks.compare_runs import compare


class MatchedRecoveryTests(unittest.TestCase):
    def checkpoint(self):
        r = promptwise(G=2, num_dim=2, rd_budget=5, model_cost=[1, 2], cost_para=.01,
                       tau_exp=1, reg_method='mle', exp_para=0)
        for g, y in [(0, 1), (1, 0)] * 10:
            r.update_stats(g, np.ones((1, 2)), y)
            r.reset_rd_stats()
        return r

    def test_identical_initial_state_then_exact_discounted_update(self):
        source = self.checkpoint()
        target = DiscountAfterFailure.from_checkpoint(source, .8, 1e6)
        x = np.ones((1, 2))
        self.assertEqual(scores(source, x), scores(target, x))
        old_gram = np.linalg.inv(source.inv_Gram_mat[0])
        target.update_stats(0, x, 0)
        expected_weights = np.r_[np.full(10, .8), 1.]
        np.testing.assert_array_equal(target.weights[0], expected_weights)
        np.testing.assert_array_equal(target.weights[1], np.ones(10))
        expected_theta = solve_lin_logistic_regression(
            target.reg_variable[0], target.reg_target[0], sample_weight=expected_weights)
        np.testing.assert_allclose(target.reg_model[0], expected_theta)
        np.testing.assert_allclose(target.grams[0], .8*old_gram+.2*1e6*np.eye(2)+x.T@x)
        self.assertEqual(len(source.reg_target[0]), 10)

    def test_unit_discount_preserves_original_feedback_updates(self):
        a = self.checkpoint()
        b = DiscountAfterFailure.from_checkpoint(a, 1., 1e6)
        for y in (0, 0, 1):
            for router in (a, b):
                router.update_stats(0, np.ones((1, 2)), y)
                router.reset_rd_stats()
            np.testing.assert_allclose(a.reg_model[0], b.reg_model[0], atol=1e-9)
            np.testing.assert_allclose(a.inv_Gram_mat[0], b.inv_Gram_mat[0], atol=1e-14)

    def test_shared_failure_and_unchosen_outcomes_do_not_set_eligibility(self):
        source = self.checkpoint()
        for other_arm in (0, 1):
            outcomes = np.zeros((2, 5), dtype=int)
            outcomes[1] = other_arm
            first, rows = evaluate_case(source, np.ones((1, 2)), outcomes, 7, 3, 15., .98, 1e6)
            self.assertEqual(first['first_failed'], 1)
            self.assertEqual(len(rows), 4)
            self.assertEqual(len({r['first_arm'] for r in rows}), 1)
            self.assertEqual(len({r['first_sample'] for r in rows}), 1)
            self.assertEqual(len({r['initial_predicted_success'] for r in rows}), 1)
            self.assertEqual(next(r for r in rows if r['policy']=='switch_on_failure')['next_action'], 'switch')
        first, rows = evaluate_case(source, np.ones((1, 2)), np.ones((2, 5), dtype=int),
                                    7, 3, 15., .98, 1e6)
        self.assertEqual(first['first_failed'], 0)
        self.assertEqual(rows, [])

    def test_no_cost_room_means_stop_including_initial_cost(self):
        source = self.checkpoint()
        first, rows = evaluate_case(source, np.ones((1, 2)), np.zeros((2, 5), dtype=int),
                                    7, 3, 1., .98, 1e6)
        self.assertEqual(first['first_failed'], 1)
        self.assertTrue(all(r['next_action']=='stop' and r['cost']==1. and r['remaining_cost']==0.
                            for r in rows))

    def test_cross_run_pairing_and_missing_episode_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for label in ('reference','candidate'):
                (root/label).mkdir()
                with (root/label/'seed_11.csv').open('w') as f:
                    w = csv.DictWriter(f,fieldnames=['history','scenario','attempt_budget','policy',
                        'mode','prompt_id','seed','success','cost'])
                    w.writeheader()
                    for q in ('a','b'):
                        w.writerow(dict(history=50,scenario='id',attempt_budget=3,policy='original',
                            mode='episodic',prompt_id=q,seed=11,success=int(label=='candidate'),cost=1))
            rows = compare(root/'reference',root/'candidate',root/'paired.csv',replicates=50)
            self.assertEqual(rows[0]['success_delta_low'],1.)
            self.assertEqual(rows[0]['cost_delta_high'],0.)
            p = root/'candidate'/'seed_11.csv'
            p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
            with self.assertRaises(ValueError):
                compare(root/'reference',root/'candidate',root/'paired.csv',replicates=50)


if __name__ == '__main__':
    unittest.main()
