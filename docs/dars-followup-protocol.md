# Follow-up: matched recovery and prior sensitivity

Protocol specified before inspecting follow-up outcomes, 16 September 2026.

## 1. Recovery after the same initial failure

Train the ORIGINAL router on the same schedule as the first pilot. Clone each
checkpoint for all four policies. The initial model, predicted probabilities,
exploration bonuses, sampled response, and incurred cost are identical. Determine
eligibility from that first response only; keep genuine failures without checking
whether an alternative model would succeed. Successful first responses and
initial stops are recorded in an eligibility table and excluded from the
conditional recovery outcome.

Policies: original; `discount_after_failure`; repeat initial; highest-ranked
untried switching. The new discount policy starts with weight 1 on EVERY historical
observation. When the first failure arrives, it multiplies old weights for the
selected arm by 0.98 and appends the new feedback with weight 1. Its Gram update
preserves the ridge. Subsequent selected-arm observations continue discounting.
Inactive arms do not age. This deliberately tests introducing forgetting after
failure, NOT a mature EWMA policy with an already discounted training history.

No policy sees unselected outcomes. Sample permutations match across policies and
caps, with no within-model sample reuse in an episode. All budgets include the
shared first call. Primary cap: 3 total calls; secondary cap: 5. Histories:
50, 200, 1000 training requests; seeds 11, 22, 33; all 60 evaluation questions;
scenarios `id` and artificial `arm_shift`; episodic resets; cost budget 15 and
cost penalty 0.05. Primary ridge: legacy 1e6; secondary matched run: ridge 1.

Report conditional recovery, total cost, the next decision (repeat/switch/stop),
the probability change after the first failure, actual observations of the first
arm, and first-arm exploration bonus. Pair policies within each eligible episode.
Bootstrap all 60 question clusters, retaining all seeds and zero-eligibility
questions; recompute the conditional denominator per bootstrap draw. Intervals
condition on the training split, seeds and stored samples. Different ridges may
produce different initial choices and failure subsets; conditional rates across
ridges are not a matched causal comparison.

## 2. End-to-end prior sensitivity

Repeat the COMPLETE original pilot with `--gram-ridge 1` instead of 1e6, holding
all other settings fixed. This changes uncertainty geometry during warm-up and
evaluation, so learned action histories can change. Compare overall outcomes on
the same questions, not just each run's own conditional failure subset. This is
a sensitivity analysis, not a claim that ridge 1 reproduces the paper or is an
optimal setting. The original quadratic (not square-root) bonus is retained.

## Commands

Use the CPU environment from the DARS replay guide, then:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m benchmarks.recovery_diagnostic \
  --data runs/dars-gpqa-v1/outcomes.npz \
  --output runs/dars-gpqa-v1/matched-recovery

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m benchmarks.recovery_diagnostic \
  --data runs/dars-gpqa-v1/outcomes.npz \
  --output runs/dars-gpqa-v1/matched-recovery-ridge1 --gram-ridge 1

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m benchmarks.retry_benchmark \
  --data runs/dars-gpqa-v1/outcomes.npz \
  --output runs/dars-gpqa-v1/replay-ridge1 \
  --histories 50 200 1000 --attempts 1 2 3 5 --seeds 11 22 33 \
  --eval-requests 60 --scenarios id arm_shift --modes episodic \
  --sampling without_replacement --discount .98 --cost-budget 15 \
  --cost-para .05 --gram-ridge 1

python -m benchmarks.summarize_paired --input runs/dars-gpqa-v1/replay-ridge1

python -m benchmarks.compare_runs \
  --reference runs/dars-gpqa-v1/replay \
  --candidate runs/dars-gpqa-v1/replay-ridge1 \
  --output runs/dars-gpqa-v1/prior-comparison.csv
```

Run heavy commands sequentially. Each uses one numerical-computation thread.
Forgetting introduced after a failure need not overcome thousands of equally
weighted prior observations within two retries. A null result is informative.

## More history versus more independent data

Repeated replay of existing records can test an accumulation mechanism, but it
cannot substantiate broad generalization from thousands of independent questions.
Keep unique question counts and replay request counts separate. Histories also
need per-arm observation counts; request count alone is not an arm's sample size.

An actionable data expansion exists in the same GPQA release: the published test
file has 248 additional questions with one original-prompt `standard` decoding
response per model at temperature 0.7/top-p 0.95/max_tokens 768. These questions
do not overlap source-training IDs. One response per model can supply training
feedback even though it is insufficient for repeated same-model evaluation.
Keeping our 60 evaluation questions fixed would increase the available training
pool from 140 to 388 unique questions. This changes the published split and mixes
original prompts with training rewrites, so it must be a separately declared
protocol with checks for text overlap, coverage, and a training schedule that
does not fabricate missing repeated responses. It is not part of the two
follow-up runs above.

For still larger-history mechanism checks, repeated exposure or weighted
replication can increase historical evidence mass while keeping question support
fixed. Label such results as empirical simulation, with fixed regularization and
documented geometry changes. Do not call replicated records new independent data
or use them to shrink uncertainty intervals. Additional question diversity and
independent repeated generations are separate needs.
