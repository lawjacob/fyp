"""Paired question-cluster bootstrap comparing two episodic replay runs."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(directory):
    groups = defaultdict(dict)
    for path in sorted(Path(directory).glob('seed_*.csv')):
        with path.open() as f:
            for r in csv.DictReader(f):
                if r['mode'] != 'episodic':
                    continue
                group = tuple(r[k] for k in ('history','scenario','attempt_budget','policy'))
                key = (r['prompt_id'], r['seed'])
                if key in groups[group]:
                    raise ValueError('Duplicate episode')
                groups[group][key] = r
    return groups


def compare(reference, candidate, output, replicates=2000):
    a, b = read(reference), read(candidate)
    if not a or a.keys() != b.keys():
        raise ValueError('Runs must have identical, nonempty settings')
    rng = np.random.default_rng(1729)
    rows = []
    for setting in sorted(a):
        if a[setting].keys() != b[setting].keys():
            raise ValueError('Runs must evaluate identical question/seed pairs')
        by_question = defaultdict(list)
        for key, ref in a[setting].items():
            cand = b[setting][key]
            by_question[key[0]].append([float(cand[k])-float(ref[k]) for k in ('success','cost')])
        values = np.array([np.mean(by_question[q], axis=0) for q in sorted(by_question)])
        draws = rng.integers(len(values), size=(replicates, len(values)))
        boot = values[draws].mean(axis=1)
        row = dict(zip(('history','scenario','attempt_budget','policy'), setting),
                   unique_questions=len(values), paired_episodes=len(a[setting]))
        for j, metric in enumerate(('success_delta','cost_delta')):
            lo, hi = np.quantile(boot[:, j], [.025,.975])
            row.update({metric: values[:, j].mean(),metric+'_low':lo,metric+'_high':hi})
        rows.append(row)
    out = Path(output)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {out}; candidate minus reference, conditional on stored data and training seeds.')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    compare(args.reference, args.candidate, args.output)


if __name__ == '__main__':
    main()
