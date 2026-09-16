# Matched recovery and Gram-prior sensitivity results

Completed 16 September 2026 using the settings specified in the
[follow-up protocol](dars-followup-protocol.md). These are follow-up diagnostics
on the same GPQA questions, not independent dataset replications or OOD proof.

## Matched recovery

All four policies start with identical coefficients, geometry, observation
history, first model, first sampled response, and first-call cost. We evaluate
only shared first failures. `discount_after_failure` starts forgetting at that
failure with gamma 0.98; historical weights were all 1 before the intervention.
It is NOT the pretrained EWMA policy in the first pilot.

Illustrative primary slice: 1000 training requests, at most three total attempts,
cost budget 15, legacy Gram ridge 1e6. Recovery is conditional on first failure;
these percentages cannot be compared directly with the pilot's overall success.

| Policy | Ordinary recovery (117 failed episodes) | Artificial-shift recovery (122 failed episodes) | Shifted mean total cost |
| --- | ---: | ---: | ---: |
| Original | 29.91% | 29.51% | 4.691 |
| Discount starting at failure | 29.91% | 29.51% | 4.709 |
| Repeat initial | 28.21% | 27.87% | 4.911 |
| Switch on failure | 47.01% | 44.26% | 6.831 |

The switching gain under shift is +14.75 percentage points, with a conditional
question-cluster bootstrap interval [+0.83, +28.82]. Mean cost rises by about
2.14 units. It does not dominate original on every objective. The discount and
original policies have identical binary recovery outcomes in this slice, yielding
a degenerate empirical bootstrap interval at zero. That is NOT proof of equality
on unseen questions or of zero possible benefit from forgetting.

With Gram ridge 1, the corresponding shifted recovery is 27.93% for original,
27.03% for discount-at-failure, and 30.63% for switching. Switching minus original
has interval [-9.65, +15.75] percentage points. This run has a different failure
subset because the prior also affects training and initial choices; do not
interpret between-ridge conditional differences as a matched intervention.

Starting forgetting late is a limited intervention: after two selected-arm
updates, old records still retain 0.98^2 = 96.04% of their initial weights.
This differs fundamentally from maintaining a discounted history throughout
training. Keeping initial states identical was necessary to isolate the late
update intervention, but it restricts the conclusion we can draw.

Probability changes after first failure also do not show a simple monotonic
history-size pattern. In the legacy-prior shifted original policy, the average
drop is about 1.13, 0.72, and 1.07 percentage points at histories 50, 200, and 1000.
The selected models/cases differ across checkpoints. A stronger causal test of
history mass needs controlled contexts, model choices, and historical evidence.

## End-to-end prior sensitivity

The new full replay differs from the first pilot ONLY in the Gram ridge (1 rather
than 1e6) and output directory. The logistic objective's regularization and the
quadratic exploration formula are unchanged. Histories, generation draws,
question split, seeds, costs and evaluation conditions match.

At history 1000 and three attempts under artificial shift:

| Policy | Legacy-prior overall success | Ridge-1 overall success | Legacy repeat rate | Ridge-1 repeat rate |
| --- | ---: | ---: | ---: | ---: |
| Original | 52.22% | 55.56% | 94.55% | 90.00% |
| Pretrained EWMA | 52.22% | 61.67% | 81.08% | 37.00% |
| Repeat initial | 51.11% | 54.44% | 100.00% | 100.00% |
| Switch on failure | 62.22% | 57.22% | 0.00% | 0.00% |

Pretrained EWMA's paired success change from the prior intervention is +9.44
percentage points, interval [+3.89, +16.11]. Mean cost falls by 0.398 units,
interval [-0.545, -0.263]. Original's success change is +3.33 points, interval
[-2.22, +8.89]. The prior materially affects this example, especially EWMA.

Within the ridge-1 run, EWMA minus original success is +6.11 points, interval
[-0.56, +13.33]. Thus this pilot does not establish a reliable general superiority
of EWMA. The all-setting tables must accompany these illustrative slices, and
the intervals are exploratory, not corrected for multiple comparisons.

## Data limitation and an available expansion

The current pilot still has 140 unique training and 60 evaluation questions.
Repeating training records increases accumulated history, not independent data.
We audited the unused GPQA source-test standard-temperature records: 248 further
questions, six models, exactly one response per question/model, with the same
temperature/top-p/token cap. There is no question-ID or normalized-original-text
overlap with source-training questions. They could expand training to 388 unique
questions while leaving our 60 evaluation questions untouched. They have NOT
been used in these follow-ups. Mixing original prompts and rewrites, reassigning
the published split, and handling one-sample training feedback require a new
explicit protocol. This is a practical next step, not a claim of thousands of
independent examples.

Large-history replay or weighted replication can separately test an accumulation
mechanism using many updates on fixed question support. Its uncertainty must be
reported conditional on the finite recorded dataset. More unique questions and
more stochastic responses per question answer different experimental needs.

## Artifacts and validation

- [Matched recovery, legacy prior](../results/dars-followup-v1/matched-summary.csv)
- [Matched recovery, ridge 1](../results/dars-followup-v1/matched-ridge1-summary.csv)
- [Full ridge-1 replay](../results/dars-followup-v1/prior-ridge1-summary.csv)
- [Paired between-prior comparisons](../results/dars-followup-v1/prior-comparison.csv)
- [Paired policy comparisons at ridge 1](../results/dars-followup-v1/prior-ridge1-paired-intervals.csv)
- [Additional-data audit](../results/dars-followup-v1/data-expansion-audit.json)
- [Resource measurements](../results/dars-followup-v1/resources.json)

Run manifests with arguments and source/data fingerprints accompany these CSVs.
Raw traces remain in the local ignored run directories. There are 17,280 new
end-to-end replay episodes, 5,544 matched episodes with the legacy prior, and
5,360 matched episodes with ridge 1. These totals reuse the same questions,
histories, seeds, and initial failures; they are not independent sample sizes.

Validation confirms matched first choices/responses/predictions/bonuses across
all four policies, exact reproduction of the corresponding original-policy
episodes from each end-to-end pilot, and compliance with hard cost/attempt caps.
Unit checks cover gamma=1 equivalence, the weighted update, feedback isolation,
budget exhaustion, and correct pairing across runs.

Measured replay wall times were 1:51, 1:21, and 1:08 (about 4 minutes 20 seconds
total), run sequentially. Peak RSS was at most approximately 105 MiB. Each process
used one numerical-computation thread and about one CPU core; no GPU inference.
These measurements exclude earlier data downloading/preparation and code tests.

The generation-truncation, scoring, finite-data, feature-representation and
code-to-paper limitations from the first pilot still apply. The useful research
direction is now to separate initial policy quality, failure-time adaptation,
and exploration geometry rather than attributing all repetition to large history.
