"""Question-cluster bootstrap for episodic paired replay, conditional on run seeds."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(directory, replicates=2000, seed=1729):
    groups = defaultdict(dict)
    for path in sorted(Path(directory).glob('seed_*.csv')):
        with path.open() as f:
            for r in csv.DictReader(f):
                if r['mode'] != 'episodic':
                    continue  # Stream feedback couples different questions.
                setting = (r['history'], r['scenario'], r['attempt_budget'])
                key = (r['prompt_id'], r['seed'], r['policy'])
                if key in groups[setting]:
                    raise ValueError('Duplicate request identity')
                groups[setting][key] = r
    if not groups:
        raise ValueError('No episodic results found')
    output = []
    rng = np.random.default_rng(seed)
    for setting, records in sorted(groups.items()):
        prompts = sorted({k[0] for k in records})
        seeds = sorted({k[1] for k in records})
        indices = rng.integers(len(prompts), size=(replicates, len(prompts)))
        for policy in ('original', 'ewma', 'repeat_initial', 'switch_on_failure'):
            values = []
            for q in prompts:
                policy_rows = [records[q, s, policy] for s in seeds]
                baseline = [records[q, s, 'original'] for s in seeds]
                # Average seeds WITHIN question; never treat reruns as new questions.
                values.append([
                    np.mean([float(r['success']) for r in policy_rows]),
                    np.mean([float(a['success'])-float(b['success']) for a, b in zip(policy_rows, baseline)]),
                    np.mean([float(a['cost'])-float(b['cost']) for a, b in zip(policy_rows, baseline)]),
                ])
            values = np.array(values)
            boot = values[indices].mean(axis=1)
            row = dict(zip(('history', 'scenario', 'attempt_budget'), setting), policy=policy,
                       unique_questions=len(prompts), seeds=len(seeds))
            for j, metric in enumerate(('success', 'success_delta_vs_original', 'cost_delta_vs_original')):
                low, high = np.quantile(boot[:, j], [.025, .975])
                row.update({metric: float(values[:, j].mean()), metric+'_low': float(low), metric+'_high': float(high)})
            output.append(row)
    target = Path(directory) / 'paired_intervals.csv'
    with target.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(output[0]))
        w.writeheader()
        w.writerows(output)
    print(f'Wrote {target}. Intervals condition on fixed training split, seeds, and recorded generations.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--replicates', type=int, default=2000)
    args = p.parse_args()
    if args.replicates < 1:
        p.error('replicates must be positive')
    summarize(args.input, args.replicates)


if __name__ == '__main__':
    main()
