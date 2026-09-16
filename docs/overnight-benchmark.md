# Overnight Qwen retry diagnostic

This implements a **new diagnostic using the original repository's linear PromptWise policy**, not a numerical reproduction of the paper. The public `test.py` generates random embeddings and outcomes. This repository contains no released Qwen outcome tables or original real-data loader. Matching the published figures requires those original datasets, model versions, embeddings, costs and settings.

## First server run

From the repository root, create an environment and install a CUDA-compatible PyTorch build for the server (follow https://pytorch.org/get-started/locally/ if it is not already installed). Then:

```bash
python -m pip install -r requirements-benchmark.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m unittest discover -s tests -v
mkdir -p runs
nohup bash scripts/overnight.sh > runs/overnight.log 2>&1 &
tail -f runs/overnight.log
```

The launch uses the active environment's `python`. Set `PYTHON=/absolute/path/to/python` if needed. On a cluster requiring a scheduler, run `bash scripts/overnight.sh` **inside an allocated GPU job**, not on the login node. No server job is launched by creating these files.

Defaults:

- Sequentially load Qwen2.5-0.5B, 1.5B and 3B Instruct on `cuda:0`; only one model is resident at a time. No quantization, CPU offloading, or multi-GPU distribution. GPU memory and runtime have not been measured on your server.
- 300 short GSM8K train prompts, 100 held-out short test prompts, 100 held-out long test prompts. The length cutoff is derived from the training split. This is a **prompt-length shift**, not a claimed semantic OOD benchmark. Both evaluation groups are disjoint from router training, but possible base-model pretraining contamination is not controlled.
- Five independently seeded generations per model/prompt: 7,500 generations total, at most 512 new tokens each. Temperature 0.7, top-p 0.9. Runtime is hardware dependent; there is no overnight completion guarantee.
- Fixed model costs 1:3:6, a rough parameter-size per-call proxy, **not actual dollars, latency, or measured FLOPs**. Total request budget 15 in these units. Generation and automated judging overhead are not included in this proxy. The collection records latency and tokens for later cost analysis.
- Warm-up checkpoints after 50, 200 and 1,000 requests, sampled with replacement from the train pool. This grows replay history; it does not create 1,000 unique training prompts. Attempts 1, 3 and 5; three routing seeds.

For a small GPU preflight in a separate output directory:

```bash
RUN_DIR=runs/qwen-preflight TRAIN_PROMPTS=6 EVAL_PROMPTS=2 SAMPLES=2 MAX_NEW_TOKENS=128 bash scripts/overnight.sh
```

This checks the actual generation pipeline before spending the full budget. Its sample sizes and token cap make it unsuitable for scientific conclusions. The default 512-token cap can also cause truncation; inspect collection statistics before interpreting quality.

## Resume and outputs

Rerun the same command to resume. Each generation is appended to a model-specific JSONL; collection skips completed prompt/sample pairs and repairs an interrupted final line. It refuses changed collection settings in the same directory. Replay restarts an incomplete seed, skips completed seeds, and refuses changed run settings. Use a new `RUN_DIR` when changing sample counts or token limits. Do not run two processes against the same directory.

Results under `runs/qwen-retry-v1/`:

- `collection/model_*.jsonl`: response text, binary label, parse status, truncation, seed, model revision, token counts and latency. Audit these first.
- `outcomes.npz`: complete model × prompt × sample outcome table and stateless 32-dimensional hashed prompt features (including intercept). These features are a diagnostic substitute, not the paper's embeddings.
- `replay/summary.csv`: success, mean cost/attempts, cost per success, repeat-after-failure rate, and recovery conditional on first-attempt failure, grouped by history, scenario, evaluation mode, attempt cap and policy. Seed SD is descriptive, **not a confidence interval**. Conditional recovery compares different subsets when policies choose different first arms.
- `replay/seed_*.csv`: every request, including ordered actions, first failure, stop reason and stream position. Use this for recovery curves and paired statistical analysis.
- `replay/traces_*.jsonl`: every selected action and observed reward, all predicted means, logit bonuses, optimistic scores before/after feedback, and evaluation-only empirical model success rates. Unchosen outcomes never enter learner updates.
- `collection/collection.json`, `replay/run.json`, `environment.txt`: settings and fingerprints.

GSM8K judging requires a final `#### number` and exact numeric equality, allowing commas and decimal formatting. It does not use an LLM judge or execute generated code. Missing answer markers count as failures; this measures format compliance as well as math accuracy. Manually audit some judgments. It is not a general math-expression grader.

## Experimental semantics

**Policies:** original linear PromptWise; exponentially weighted linear PromptWise; repeat the initial original-router choice; switch to the highest-ranked untried model after failure. Baselines use the original first-choice and learning machinery; repeat/switch override subsequent selection and do not adopt subsequent utility-based stopping. Switch stops after exhausting the model pool. All stop at success, attempt limit or hard cost limit. To preserve the original selection rule, a selected unaffordable action causes a stop rather than reranking the affordable subset. Stop reasons are logged.

**Modes:** episodic restores the trained checkpoint before each evaluation request, isolating within-request correction; stream retains feedback between evaluation requests. Original, repeat and switch start from the same original checkpoint. EWMA has its own discounted warm-up checkpoint. Consequently the EWMA comparison includes both initial-prior differences and retry adaptation, not solely a change introduced after first failure.

**Scenarios:** `id` and `length_shift` use their separate held-out pools. `arm_shift` uses the ID prompts and cyclically reassigns model outcome tables while retaining costs, starting immediately after warm-up. This is an artificial intervention on conditional rewards, not evidence of a real Qwen update. The benchmark does not yet measure retention on interleaved unaffected contexts.

Replay samples with replacement from a finite outcome table. Each model's kth attempt uses the same indexed random draw across policies and attempt caps for a given request/seed. Only selected feedback is revealed. Five samples give coarse empirical success estimates; replay seeds do not create additional independent model generations. Request-level/seed results should not be treated as independent draws from a large population without accounting for shared prompts and generation samples.

**Discount definition:** after the nth observation of arm g, its logistic loss weights are `gamma ** (n-j)` for observation j (1-indexed). L2 regularization remains fixed. The Gram matrix is `lambda I + sum_j gamma**(n-j) x_j x_j^T`. Discounting uses an **arm-observation clock**: an unselected arm does not age. This is a deliberate simple baseline, not a global-time discounted-UCB implementation or a claim of non-stationary regret guarantees. Default gamma=0.98. Gamma=1 recovers original linear fitting.

The original repository uses initial inverse Gram `1e-6 I` (Gram prior `1e6 I`) and a quadratic exploration bonus rather than its square root. Both are retained by default. A separately labeled `--gram-ridge 1` run can diagnose sensitivity to this prior; it changes all compared policies consistently. Kernel logistic regression is not implemented for discounting in this runner. The legacy `test.py` remains available for its original KLR/random-data example.

## More runs without more GPU generation

```bash
python -m benchmarks.retry_benchmark --data runs/qwen-retry-v1/outcomes.npz --output runs/qwen-retry-v1/replay-gamma95 --discount .95
python -m benchmarks.retry_benchmark --data runs/qwen-retry-v1/outcomes.npz --output runs/qwen-retry-v1/replay-ridge1 --gram-ridge 1
```

For custom model pools, run `python -m benchmarks.collect_qwen collect --help` and use `--models ...` in the manual collect command. Then pack with matching `--costs ...`. For real OOD data, supply a JSONL with unique `id`, `prompt`, `answer` (numeric `####` format), and `split` values `train`, `id`, and e.g. `ood`; replay with `--scenarios id ood arm_shift`. Other tasks may supply an NPZ directly with `contexts[N,D]`, binary `outcomes[G,N,S]`, positive `costs[G]`, unique `ids[N]`, and `splits[N]`. T2I requires its own collector/judge but can reuse the replay runner.

The initial question is whether repetition loses useful success opportunities, and whether its relationship to history size persists across seeds. A large repetition rate alone is not proof of bad routing.
