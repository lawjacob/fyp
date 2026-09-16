# DARS empirical retry pilot

This runs four policies on stored outputs from six models. It requires CPU scientific Python only; no model weights, Torch, GPU, API key, or inference charges.

## Four policies

| Policy | Behaviour after failure |
| --- | --- |
| `original` | The repository's linear PromptWise updates from the selected outcome and chooses again. |
| `ewma` | The same selection machinery with exponentially weighted logistic fitting and Gram statistics; gamma 0.98 per selected-arm observation. |
| `repeat_initial` | Uses the original router's first choice and keeps requesting that model. |
| `switch_on_failure` | Uses the original first choice, then the highest-ranked untried model according to the updated router. |

These are four policies, evaluated under common experimental conditions. Repeat/switch override subsequent utility-based stopping, while retaining hard cost and attempt limits. An unaffordable selected action stops the request; the runner does not rerank affordable arms. All policies stop on success. Original/repeat/switch share their warm-up checkpoint; EWMA has its own. Thus EWMA measures an end-to-end discounted policy, not an intervention exclusively after the first failure.

## Important correction from inspecting the files

The published GPQA test file has three `decoding_variation` observations, but they use different temperatures (`low_temp`, `standard`, `high_temp`). They cannot be treated as three repeats of one fixed configuration. Earlier notes based on the paper's description overstated this.

The importer instead uses source-training rewrite 0, which has five decodes per question/model with identical request settings. It creates a NEW parent-question split: 140 training and 60 held-out questions from the 200 source-training questions. All other rewrites and the published test file are excluded. The same exact rewritten question and answer choices are supplied to every model. The adapter checks shared prompt text, all six models, all five decoding IDs, duplicate generation/response IDs, binary scores, and positive costs. It rejects incomplete coverage instead of treating missing outcomes as failures.

The checked GPQA slice contains 6,000 responses (200 questions × six models × five samples), at temperature 0.7, top-p 0.95, max output 768 tokens. Its Qwen responses have 70% `finish_reason=length`, and Gemini has 80%. Retain these failures and disclose that the experiment studies the released generation configurations. It is not a ranking of unrestricted model capability. Scores are the published labels, not independently verified ground truth; the source's `parse_success` flag can be true even when `predicted_answer` is missing. Inspect both fields. The GSM8K judge is unchanged and unused here.

## Install and prepare

From the project root on your own machine, using Python 3.10 or newer:

```bash
python3 -m venv .venv-replay
source .venv-replay/bin/activate
python -m pip install -r requirements-replay.txt
python -m unittest tests.test_retry_benchmark tests.test_dars -v
python -m benchmarks.prepare_dars --dataset gpqa --output runs/dars-gpqa-v1
```

The preparer downloads approximately 104 MB of JSONL from a pinned dataset commit. Its reduced arrays are small. Read `runs/dars-gpqa-v1/audit.json` before interpreting results. To use an existing download, add `--source-file <training_jsonl>`. Its actual SHA256 is recorded; verify provenance when providing your own file. Existing packed outputs are protected against overwriting.

## First empirical run

```bash
python -m benchmarks.retry_benchmark \
  --data runs/dars-gpqa-v1/outcomes.npz \
  --output runs/dars-gpqa-v1/replay \
  --histories 50 200 1000 --attempts 1 2 3 5 \
  --seeds 11 22 33 --eval-requests 60 \
  --scenarios id arm_shift --modes episodic \
  --sampling without_replacement \
  --discount 0.98 --cost-budget 15 --cost-para 0.05
python -m benchmarks.summarize_paired --input runs/dars-gpqa-v1/replay
```

The runner executes all four policies automatically. No separate policy command is required. For a plumbing check use a separate output directory with `--histories 12 --attempts 1 3 --seeds 11 --eval-requests 4 --scenarios id`; do not present that smoke run as evidence.

`id` is the random held-out split of the fixed-rewrite population. `arm_shift` cyclically reassigns model outcome tables after warm-up while keeping costs fixed: an explicitly artificial change in success behaviour. This first run does NOT include a held-out-domain experiment. Repeating it on MATH is a second task, not automatically OOD evaluation. A domain experiment needs training on one domain and evaluating on another with compatible features, costs, and model settings.

Each episode draws stored samples WITHOUT replacement within each model, with the same sample order across policies and attempt caps. Across episodes, seeds, and warm-up requests, records can be revisited. This increases simulation history, not the number of independent generations or unique training questions. Evaluation restores a checkpoint per question; use stream mode only as a separately reported experiment.

Costs are fixed means of raw recorded `usage.cost` over the NEW training split, divided by the cheapest model's mean. They are expected-cost proxies, not each response's realized dollars or a guarantee on actual token expenditure. Every failed call is charged. The 15-unit cap is declared in advance; add separate budget sensitivity runs before claiming a general cost advantage. Do not tune settings against this held-out split.

## Outputs and interpretation

- `audit.json`: provenance, coverage, settings, costs, missing predictions, and truncation counts.
- `replay/summary.csv`: success, mean cost, attempts, repeat rate, and conditional recovery for each setting/policy.
- `replay/paired_intervals.csv`: success and paired differences versus original, with 95% question-cluster bootstrap intervals. Seeds are averaged within each question; intervals condition on this training split, seeds, and recorded samples. They do not capture all training uncertainty or live model variation.
- `replay/seed_*.csv`: request results, action sequences, and stop reasons.
- `replay/traces_*.jsonl`: predictions, exploration bonuses, selected sample, observed feedback, and cost per attempt. The next decision is the next trace row; stopping is in the request CSV. Counterfactual rates in traces are evaluation-only and never used by the router.

Conditioning recovery on first failure selects different populations when policies choose different first models. Prefer the paired overall success/cost comparisons for the main claim. Cost per success includes expenditure on unsuccessful requests; it is not merely the average cost among solved questions.

The original repository's inverse Gram prior `1e-6 I` and quadratic exploration formula are preserved. Hashed 32-dimensional features replace the paper's embeddings. Therefore this is an empirical diagnostic using the original implemented policy, NOT an exact reproduction of paper results. A separately labelled `--gram-ridge 1` sensitivity run helps distinguish large-history behaviour from a very strong initial prior. EWMA discounts logistic observations, not an arithmetic mean of success rates.

For a second task, prepare `--dataset math-500 --output runs/dars-math-v1` and run the same commands using that directory. The verified first-record schema uses binary exact-match labels and max output 1536 tokens; the importer will audit the entire file at preparation time. This path has not been validated end to end on the full MATH release yet. More tasks help robustness; no result should be presented as proof that EWMA must outperform original PromptWise.

Sources: [DARS dataset](https://huggingface.co/datasets/AIGNLAI/DARS), pinned commit `3109e08fb15b24192c0284b9fc11ca9f71110950`; [paper](https://arxiv.org/abs/2606.06924).
