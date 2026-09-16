# First DARS GPQA replay pilot

Completed 16 September 2026 using the [documented protocol](dars-replay.md).
The source-training file SHA256 is `9a9f9c797647b8397a6b4eee2ce9915187dcf8dd5a3a271e8b33f4abec006025`.

Checked-in artifacts: [all-setting summaries](../results/dars-gpqa-v1/summary.csv),
[paired intervals](../results/dars-gpqa-v1/paired_intervals.csv),
[data audit](../results/dars-gpqa-v1/audit.json), and
[replay configuration and fingerprints](../results/dars-gpqa-v1/run.json).
Raw responses, packed arrays, and request traces are not included in the repository.

The pilot uses 200 unique questions, freshly split into 140 training and 60 evaluation questions (split seed 42), six models, and five recorded decodes per fixed rewrite. It evaluates four policies, three histories (50/200/1000), four attempt caps (1/2/3/5), two scenarios, and three routing seeds (11/22/33): 17,280 replayed evaluation requests. These are repeated uses of 60 held-out questions, NOT 17,280 independent questions or new model calls.

The following is an illustrative slice: history 1000, at most three attempts, expected-cost proxy budget 15, episodic mode. Inspect the complete CSVs for all histories and caps.

| Policy | Held-out success | Held-out mean cost | Artificial-shift success | Artificial-shift mean cost |
| --- | ---: | ---: | ---: | ---: |
| Original PromptWise | 54.44% | 3.257 | 52.22% | 3.608 |
| EWMA PromptWise | 67.22% | 2.617 | 52.22% | 3.047 |
| Repeat initial | 53.33% | 3.373 | 51.11% | 3.757 |
| Switch on failure | 65.56% | 4.755 | 62.22% | 5.058 |

Question-cluster bootstrap intervals (95%, 2000 replicates, conditional on the fixed training split, seeds and recorded generations):

- EWMA minus original success: held-out +12.78 percentage points, interval [+5.00, +20.56]; artificial shift 0.00 points, interval [-6.67, +6.12].
- Switching minus original success under artificial shift: +10.00 points, interval [+1.11, +18.89], with greater mean cost. This is not evidence of uniform cost dominance.

Original repeated the same model on 95.65% of executed retry transitions in the held-out slice and 94.55% under artificial shift. These rates exclude failures after which no retry occurred. Repetition alone does not establish an error; the paired success/cost comparisons are necessary context.

This supports further investigation of failure recovery. It does not establish that EWMA generally fixes distribution shift: EWMA has a separately trained initial checkpoint, the shift is a constructed permutation, the evaluation set is small, multiple settings were evaluated, and truncated generations are common (Qwen 70%, Gemini 80%). The legacy strong Gram prior and hashing features also limit interpretation. See the protocol for full caveats; no settings were retuned after viewing these results.

Outputs are in the local ignored directory `runs/dars-gpqa-v1/`: audit, packed arrays, prompt split, replay manifest, request CSVs, traces, summary, and paired intervals. Reproducible commands are in the protocol guide. Validation: 12 CPU tests passed; all three completed seed files contain 5760 requests; all request costs and attempts respect caps; original/repeat/switch share first choices for every paired request. The cluster cache-lock test is outside this CPU-only path and was not rerun.

Test environment: Python 3.10, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2; one OpenBLAS/OpenMP thread during the pilot. No GPU inference was used.
