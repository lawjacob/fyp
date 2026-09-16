"""Matched-state recovery: discounting starts at the shared first failure.

This is not the separately pretrained EWMA policy in retry_benchmark.
Every policy has identical history, coefficients, geometry, first action and
first sampled response. Eligibility depends only on that first response.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from algorithms.promptwise import promptwise, solve_lin_logistic_regression
from benchmarks.collect_qwen import stable_seed
from benchmarks.retry_benchmark import load_data, make_router, request

POLICIES = ('original', 'discount_after_failure', 'repeat_initial', 'switch_on_failure')


class DiscountAfterFailure(promptwise):
    """Initially identical to an undiscounted checkpoint; old weights start at 1."""
    @classmethod
    def from_checkpoint(cls, source, discount, gram_ridge):
        if source.reg_method != 'mle' or source.per_step_update or source.rd_used_budget:
            raise ValueError('Require an idle per-observation linear MLE checkpoint')
        if not 0 < discount <= 1 or gram_ridge <= 0 or np.any(source.visitation == 0):
            raise ValueError('Require valid discount/ridge and observations of every arm')
        target = cls.__new__(cls)
        target.__dict__ = copy.deepcopy(source.__dict__)
        target.discount = discount
        target.gram_ridge = gram_ridge
        target.weights = [np.ones(len(y)) for y in source.reg_target]
        target.grams = [np.linalg.inv(v) for v in source.inv_Gram_mat]
        target.forgetting_started = False
        return target

    def fit_reg_model(self, g):
        self.reg_model[g] = solve_lin_logistic_regression(
            self.reg_variable[g], self.reg_target[g], sample_weight=self.weights[g])

    def update_stats(self, g, context, reward):
        if self.rd_skip:
            return super().update_stats(g, context, reward)
        # No forgetting on a successful first response. The experiment evaluates
        # only genuine first failures drawn before looking at alternative outcomes.
        self.forgetting_started = self.forgetting_started or reward == 0
        gamma = self.discount if self.forgetting_started else 1.
        self.weights[g] = np.append(gamma * self.weights[g], 1.)
        self.grams[g] = gamma * self.grams[g] + (1-gamma)*self.gram_ridge*np.eye(self.num_dim)
        self.inv_Gram_mat[g] = np.linalg.inv(self.grams[g])
        super().update_stats(g, context, reward)
        self.grams[g] += context.T @ context


def evaluate_case(source, x, rewards, seed, attempts, budget, discount, ridge, trace=None, tags=None):
    first = request(copy.deepcopy(source), x, rewards, 'original', seed, 1, budget,
                    sampling='without_replacement')
    if first['first_failed'] != 1:
        return first, []
    rows = []
    for policy in POLICIES:
        router = (DiscountAfterFailure.from_checkpoint(source, discount, ridge)
                  if policy == 'discount_after_failure' else copy.deepcopy(source))
        buffer = io.StringIO()
        row = request(router, x, rewards, policy, seed, attempts, budget, buffer,
                      dict(tags or {}, policy=policy), sampling='without_replacement')
        events = [json.loads(line) for line in buffer.getvalue().splitlines()]
        assert row['first_arm'] == first['first_arm'] and row['first_failed'] == 1
        g = row['first_arm']
        event = events[0]
        row.update(policy=policy, first_sample=event['sample'],
                   initial_predicted_success=event['predicted_success'][g],
                   initial_logit_bonus=event['logit_bonus'][g],
                   probability_drop_after_failure=(event['predicted_success'][g]
                                                  - event['after']['predicted_success'][g]),
                   first_arm_history_observations=int(source.visitation[g]),
                   next_action=('stop' if len(events) < 2 else
                                'repeat' if events[1]['arm'] == g else 'switch'),
                   remaining_cost=budget-float(source.model_cost[g]))
        if rows:
            assert row['first_sample'] == rows[0]['first_sample']
            assert row['initial_predicted_success'] == rows[0]['initial_predicted_success']
            assert row['initial_logit_bonus'] == rows[0]['initial_logit_bonus']
        rows.append(dict(tags or {}, **row))
        if trace is not None:
            trace.write(buffer.getvalue())
    return first, rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def summarize(rows, eligibility, out, replicates=2000):
    groups, eligible_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[(row['history'], row['scenario'], row['attempt_budget'])].append(row)
    for row in eligibility:
        eligible_groups[(row['history'], row['scenario'])].append(row)
    summary = []
    rng = np.random.default_rng(1729)
    for setting, group in sorted(groups.items()):
        population = eligible_groups[setting[:2]]
        # Bootstrap ALL evaluated questions, retaining zero eligible episodes.
        # Pair seeds and all policies within question; don't count repeated seeds
        # as independent questions. Conditional denominator is recomputed per draw.
        ids = sorted({r['prompt_id'] for r in population})
        lookup = {q: i for i, q in enumerate(ids)}
        indices = rng.integers(len(ids), size=(replicates, len(ids)))
        baseline = {(r['prompt_id'], r['seed']): r for r in group if r['policy'] == 'original'}
        for policy in POLICIES:
            selected = [r for r in group if r['policy'] == policy]
            aggregates = np.zeros((len(ids), 5))
            for r in selected:
                b = baseline[r['prompt_id'], r['seed']]
                aggregates[lookup[r['prompt_id']]] += [1, r['success'], r['cost'],
                    r['success']-b['success'], r['cost']-b['cost']]
            totals = aggregates.sum(axis=0)
            boot = aggregates[indices].sum(axis=1)
            valid = boot[:, 0] > 0
            stats = boot[valid, 1:] / boot[valid, :1]
            result = dict(history=setting[0], scenario=setting[1], attempt_budget=setting[2],
                          policy=policy, eligible_failures=len(selected),
                          distinct_failed_questions=len({r['prompt_id'] for r in selected}),
                          evaluated_questions=len(ids), evaluated_episodes=len(population),
                          next_repeat_rate=np.mean([r['next_action']=='repeat' for r in selected]),
                          next_switch_rate=np.mean([r['next_action']=='switch' for r in selected]),
                          next_stop_rate=np.mean([r['next_action']=='stop' for r in selected]),
                          mean_probability_drop=np.mean([r['probability_drop_after_failure'] for r in selected]),
                          mean_first_arm_observations=np.mean([r['first_arm_history_observations'] for r in selected]),
                          mean_initial_logit_bonus=np.mean([r['initial_logit_bonus'] for r in selected]),
                          valid_bootstrap_draws=int(valid.sum()))
            for i, metric in enumerate(('recovery', 'mean_cost', 'recovery_delta', 'cost_delta')):
                lo, hi = np.quantile(stats[:, i], [.025, .975])
                result.update({metric: totals[i+1]/totals[0], metric+'_low': lo, metric+'_high': hi})
            summary.append(result)
    write_csv(Path(out)/'summary.csv', summary)


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'run.json').exists():
        raise ValueError('Use a fresh output directory for this diagnostic')
    data = load_data(args.data)
    if max(args.attempts) > data['outcomes'].shape[2]:
        raise ValueError('Attempt cap exceeds available recorded samples')
    config = vars(args).copy()
    config.update(protocol='Identical undiscounted checkpoint; discounting starts at first failure; '
                           'condition only on the shared first response. No retroactive age weighting.',
                  data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                  source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                     ['benchmarks/recovery_diagnostic.py', 'benchmarks/retry_benchmark.py',
                      'benchmarks/routers.py', 'algorithms/promptwise.py']})
    (out/'run.json').write_text(json.dumps(config, indent=2)+'\n')
    train = np.flatnonzero(data['splits']=='train')
    pool = np.flatnonzero(data['splits']=='id')
    rows, eligibility = [], []
    for seed in args.seeds:
        schedule = np.random.default_rng(seed).choice(train, max(args.histories), replace=True)
        router = make_router(data, args)
        checkpoints = {}
        for t, i in enumerate(schedule):
            request(router, data['contexts'][i:i+1], data['outcomes'][:, i], 'original',
                    stable_seed(seed, 'train', t), max(args.attempts), args.cost_budget,
                    sampling='without_replacement')
            if t+1 in args.histories:
                if np.any(router.visitation == 0):
                    raise ValueError('Every arm must be visited before saving a checkpoint')
                checkpoints[t+1] = copy.deepcopy(router)
                print(f'Seed {seed}: shared warm-up {t+1}, visits={router.visitation.tolist()}', flush=True)
        ids = np.random.default_rng(stable_seed(seed, 'eval', 'id')).permutation(pool)[:args.eval_requests]
        seed_rows, seed_eligibility = [], []
        with (out/f'traces_{seed}.jsonl').open('w', buffering=1) as trace:
            for history, source in checkpoints.items():
                for scenario in args.scenarios:
                    for t, i in enumerate(ids):
                        rewards = data['outcomes'][:, i]
                        if scenario == 'arm_shift':
                            rewards = np.roll(rewards, 1, axis=0)
                        tags = dict(seed=seed, history=history, scenario=scenario,
                                    prompt_id=str(data['ids'][i]), request=t)
                        for cap_index, attempts in enumerate(args.attempts):
                            first, results = evaluate_case(source, data['contexts'][i:i+1], rewards,
                                stable_seed(seed, 'eval', 'id', t), attempts, args.cost_budget,
                                args.discount, args.gram_ridge, trace,
                                dict(tags, attempt_budget=attempts))
                            if cap_index == 0:
                                seed_eligibility.append(dict(tags, first_arm=first['first_arm'],
                                    first_failed=first['first_failed'], eligible=int(first['first_failed']==1)))
                            seed_rows.extend(results)
        write_csv(out/f'seed_{seed}.csv', seed_rows)
        write_csv(out/f'eligibility_{seed}.csv', seed_eligibility)
        rows.extend(seed_rows)
        eligibility.extend(seed_eligibility)
        print(f'Seed {seed} complete: {len(seed_rows)} matched recovery episodes', flush=True)
    summarize(rows, eligibility, out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--histories', nargs='+', type=int, default=[50, 200, 1000])
    p.add_argument('--attempts', nargs='+', type=int, default=[3, 5])
    p.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    p.add_argument('--scenarios', nargs='+', choices=['id', 'arm_shift'], default=['id', 'arm_shift'])
    p.add_argument('--eval-requests', type=int, default=60)
    p.add_argument('--discount', type=float, default=.98)
    p.add_argument('--gram-ridge', type=float, default=1e6)
    p.add_argument('--cost-para', type=float, default=.05)
    p.add_argument('--cost-budget', type=float, default=15.)
    p.add_argument('--exp-para', type=float, default=None)
    args = p.parse_args()
    if (min(args.histories)<6 or min(args.attempts)<2 or args.eval_requests<1
            or not 0 < args.discount <= 1 or not np.isfinite(args.gram_ridge)
            or args.gram_ridge<=0 or not np.isfinite(args.cost_budget) or args.cost_budget<=0
            or not np.isfinite(args.cost_para) or args.cost_para<0):
        p.error('Invalid diagnostic configuration')
    run(args)


if __name__ == '__main__':
    main()
