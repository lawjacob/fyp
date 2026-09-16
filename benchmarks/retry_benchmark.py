"""Paired replay benchmark; learners observe only selected-arm binary feedback."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from algorithms.promptwise import promptwise
from benchmarks.collect_qwen import stable_seed
from benchmarks.routers import DiscountedPromptWise, scores


def load_data(path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    x, y, c, split = (data[k] for k in ('contexts', 'outcomes', 'costs', 'splits'))
    if (x.ndim != 2 or y.ndim != 3 or y.shape[1] != len(x) or y.shape[0] != len(c)
            or len(split) != len(x) or y.shape[2] == 0 or len(c) < 2
            or not np.all(np.isin(y, [0, 1])) or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(c)) or np.any(c <= 0)):
        raise ValueError('Invalid contexts/outcomes/costs/splits in NPZ')
    if len(set(data['ids'].tolist())) != len(x):
        raise ValueError('Prompt IDs must be unique')
    if not np.any(split == 'train') or not np.any(split == 'id'):
        raise ValueError('Require nonempty train and id splits')
    return data


def synthetic(path, seed=42):
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(size=(160, 7)), np.ones(160)])
    theta = rng.normal(size=(3, 8))
    p = 0.05 + 0.9 / (1 + np.exp(-(theta @ x.T)))
    y = (rng.random((3, 160, 12)) < p[:, :, None]).astype(np.int8)
    np.savez_compressed(path, contexts=x, outcomes=y, costs=[1., 2., 3.],
                        ids=[f'synthetic-{i}' for i in range(160)],
                        splits=['train']*100 + ['id']*30 + ['length_shift']*30,
                        models=['synthetic-a', 'synthetic-b', 'synthetic-c'],
                        metadata='Synthetic smoke fixture; not evidence about Qwen or OOD.')


def make_router(data, args, discounted=False):
    kwargs = dict(G=len(data['costs']), num_dim=data['contexts'].shape[1],
                  rd_budget=max(args.attempts), model_cost=data['costs'], cost_para=args.cost_para,
                  tau_exp=1, reg_method='mle', kernel_method='lin', exp_para=args.exp_para)
    if discounted:
        return DiscountedPromptWise(**kwargs, discount=args.discount, gram_ridge=args.gram_ridge)
    r = promptwise(**kwargs)
    r.inv_Gram_mat = [np.eye(r.num_dim)/args.gram_ridge for _ in range(r.G)]
    return r


def request(router, x, rewards, policy, seed, max_attempts, budget, writer=None, tags=None,
            sampling='with_replacement'):
    """Independent RNG for outcomes; arm's kth pull has the same draw across policies."""
    np.random.seed(stable_seed(seed, 'choices'))
    if sampling not in ('with_replacement', 'without_replacement'):
        raise ValueError('Unknown sampling mode')
    if sampling == 'without_replacement' and max_attempts > rewards.shape[1]:
        raise ValueError('Attempt cap exceeds independent recorded samples per model')
    draws = np.array([
        (np.random.default_rng(stable_seed(seed, 'outcomes', g)).permutation(rewards.shape[1])[:max_attempts]
         if sampling == 'without_replacement' else
         np.random.default_rng(stable_seed(seed, 'outcomes', g)).integers(rewards.shape[1], size=max_attempts))
        for g in range(router.G)])
    router.rd_budget = max_attempts
    actions, visits, cost, success = [], np.zeros(router.G, dtype=int), 0., 0
    first = None
    first_failed = None
    reason = 'attempt_budget'
    repeat, opportunities = 0, 0
    for attempt in range(max_attempts):
        before = scores(router, x)
        if policy == 'repeat_initial' and first is not None:
            arm = first
        elif policy == 'switch_on_failure' and first is not None:
            candidates = [g for g in range(router.G) if g not in actions]
            if not candidates:
                reason = 'all_models_tried'
                break
            q = before['optimistic_success']
            # Exploration fallback matters only for unusually tiny warm-up histories.
            arm = max(candidates, key=lambda g: float('inf') if q[g] is None else
                      1 - router.cost_para*router.model_cost[g]/max(q[g], 1e-15))
        else:
            arm = router.select_arm(x)
        if arm is None:
            reason = 'router_stop'
            break
        arm = int(arm)
        if cost + router.model_cost[arm] > budget + 1e-10:
            reason = 'cost_budget'
            break
        if first is None:
            first = arm
        if actions:
            opportunities += 1
            repeat += int(arm == actions[-1])
        sample = int(draws[arm, visits[arm]])
        reward = int(rewards[arm, sample])
        if first_failed is None:
            first_failed = 1 - reward
        visits[arm] += 1
        cost += float(router.model_cost[arm])
        actions.append(arm)
        router.update_stats(arm, x, reward)
        if writer is not None:
            row = dict(tags or {}, attempt=attempt+1, arm=arm, reward=reward, sample=sample,
                       cumulative_cost=cost, **before, after=scores(router, x),
                       # Evaluation-only; never used to select an action.
                       empirical_arm_success=rewards.mean(axis=1).tolist())
            writer.write(json.dumps(row, allow_nan=False)+'\n')
        success = reward
        if reward:
            reason = 'success'
            break
    router.reset_rd_stats()
    return dict(success=success, cost=cost, attempts=len(actions), first_arm=first,
                first_failed=first_failed,
                repeats=repeat, retry_opportunities=opportunities, stop_reason=reason,
                actions=json.dumps(actions))


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(args.data)
    if args.sampling == 'without_replacement' and max(args.attempts) > data['outcomes'].shape[2]:
        raise ValueError('Attempt cap exceeds recorded samples per model')
    if min(args.histories) < len(data['costs']):
        raise ValueError('Smallest history must allow at least one exploration request per model')
    config = vars(args).copy()
    config['data_sha256'] = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    root = Path(__file__).resolve().parents[1]
    config['source_sha256'] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                              for name in ['benchmarks/retry_benchmark.py', 'benchmarks/routers.py',
                                           'benchmarks/collect_qwen.py', 'algorithms/promptwise.py']}
    config['cost_note'] = 'Fixed costs from NPZ; inspect its metadata for units. No wall-clock/dollar claim.'
    manifest = out/'run.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Run configuration differs; use a new output directory')
    manifest.write_text(json.dumps(config, indent=2))
    train = np.flatnonzero(data['splits'] == 'train')
    all_results = []
    for seed in args.seeds:
        completed = out/f'seed_{seed}.csv'
        if completed.exists():
            with completed.open() as f:
                all_results.extend(list(csv.DictReader(f)))
            print(f'Seed {seed} already complete', flush=True)
            continue
        checkpoints = {}
        schedule = np.random.default_rng(seed).choice(train, max(args.histories), replace=True)
        for name in ('original', 'ewma'):
            router = make_router(data, args, name == 'ewma')
            for t, i in enumerate(schedule):
                request(router, data['contexts'][i:i+1], data['outcomes'][:, i], 'original',
                        stable_seed(seed, 'train', t), max(args.attempts), args.cost_budget,
                        sampling=args.sampling)
                if t+1 in args.histories:
                    checkpoints[name, t+1] = copy.deepcopy(router)
                    print(f'Seed {seed}: {name} warm-up {t+1} requests, arm visits={router.visitation.tolist()}', flush=True)
            if np.any(router.visitation == 0):
                raise ValueError('Warm-up did not visit every arm; increase cost budget')
        rows = []
        # Restart incomplete seeds deterministically; completed seeds are skipped.
        with (out/f'traces_{seed}.jsonl').open('w', buffering=1) as trace:
            for history in args.histories:
                for scenario in args.scenarios:
                    split = 'id' if scenario == 'arm_shift' else scenario
                    pool = np.flatnonzero(data['splits'] == split)
                    if not len(pool):
                        raise ValueError(f'Missing evaluation split: {split}')
                    ids = np.random.default_rng(stable_seed(seed, 'eval', split)).permutation(pool)[:args.eval_requests]
                    for mode in args.modes:
                        for attempts in args.attempts:
                            for policy in ('original', 'ewma', 'repeat_initial', 'switch_on_failure'):
                                source = checkpoints['ewma' if policy == 'ewma' else 'original', history]
                                router = copy.deepcopy(source)
                                for t, i in enumerate(ids):
                                    if mode == 'episodic':
                                        router = copy.deepcopy(source)
                                    rewards = data['outcomes'][:, i]
                                    if scenario == 'arm_shift':
                                        # Controlled intervention; not a real model update or natural OOD.
                                        rewards = np.roll(rewards, 1, axis=0)
                                    tags = dict(seed=seed, history=history, scenario=scenario, mode=mode,
                                                attempt_budget=attempts, policy=policy, request=t,
                                                prompt_id=str(data['ids'][i]))
                                    result = request(router, data['contexts'][i:i+1], rewards, policy,
                                                     stable_seed(seed, 'eval', split, t), attempts,
                                                     args.cost_budget, trace if args.trace else None, tags,
                                                     sampling=args.sampling)
                                    rows.append(dict(tags, **result))
        temp = completed.with_suffix('.tmp')
        with temp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(completed)
        all_results.extend(rows)
        summarize(all_results, out)
        print(f'Seed {seed} complete: {len(rows)} evaluated requests', flush=True)
    summarize(all_results, out)


def summarize(rows, out):
    groups = {}
    keys = ['history', 'scenario', 'mode', 'attempt_budget', 'policy']
    for row in rows:
        key = tuple(str(row[k]) for k in keys)
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group in groups.items():
        seed_rates = [np.mean([float(r['success']) for r in group if str(r['seed']) == seed])
                      for seed in sorted({str(r['seed']) for r in group})]
        successes = sum(float(r['success']) for r in group)
        retries = sum(int(r['retry_opportunities']) for r in group)
        failed_first = [r for r in group if str(r['first_failed']) == '1']
        summaries.append(dict(zip(keys, key), requests=len(group), seeds=len(seed_rates),
                              success_rate=successes/len(group),
                              success_rate_seed_sd=float(np.std(seed_rates, ddof=1)) if len(seed_rates)>1 else '',
                              mean_cost=np.mean([float(r['cost']) for r in group]),
                              mean_attempts=np.mean([int(r['attempts']) for r in group]),
                              cost_per_success=sum(float(r['cost']) for r in group)/successes if successes else '',
                              repeat_after_failure=sum(int(r['repeats']) for r in group)/retries if retries else '',
                              retry_opportunities=retries))
        summaries[-1]['first_failures'] = len(failed_first)
        summaries[-1]['recovery_given_first_failure'] = (
            np.mean([float(r['success']) for r in failed_first]) if failed_first else '')
    with (out/'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--synthetic', action='store_true', help='Create synthetic data if absent; smoke checks only')
    p.add_argument('--histories', nargs='+', type=int, default=[50, 200, 1000])
    p.add_argument('--attempts', nargs='+', type=int, default=[1, 3, 5])
    p.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    p.add_argument('--scenarios', nargs='+', default=['id', 'length_shift', 'arm_shift'])
    p.add_argument('--modes', nargs='+', choices=['episodic', 'stream'], default=['episodic', 'stream'])
    p.add_argument('--eval-requests', type=int, default=100)
    p.add_argument('--discount', type=float, default=0.98)
    p.add_argument('--gram-ridge', type=float, default=1e6, help='Legacy repo prior; use 1 as a separate sensitivity run')
    p.add_argument('--cost-para', type=float, default=0.05)
    p.add_argument('--cost-budget', type=float, default=15.)
    p.add_argument('--exp-para', type=float, default=None)
    p.add_argument('--trace', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--sampling', choices=['with_replacement', 'without_replacement'],
                   default='with_replacement', help='Use without_replacement for finite empirical retry traces')
    args = p.parse_args()
    if (min(args.histories+args.attempts) < 1 or args.eval_requests < 1
            or not np.isfinite(args.cost_budget) or args.cost_budget <= 0
            or not np.isfinite(args.cost_para) or args.cost_para < 0
            or not np.isfinite(args.gram_ridge) or args.gram_ridge <= 0
            or not 0 < args.discount <= 1):
        p.error('Invalid history, attempts, budget, regularization, or discount')
    if args.synthetic and not Path(args.data).exists():
        Path(args.data).parent.mkdir(parents=True, exist_ok=True)
        synthetic(args.data)
    run(args)


if __name__ == '__main__':
    main()
