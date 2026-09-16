# Empirical replay direction

Recorded 16 September 2026. The user wants more realistic evidence than hand-selected Bernoulli success probabilities, with a local 16 GB machine and unreliable cluster access.

## Current decision

Use public, precomputed, repeated LLM responses for a CPU-based routing/retry pilot. Keep controlled synthetic experiments as mechanism tests. The user subsequently authorized implementation. See [the runnable DARS protocol](dars-replay.md). Data coverage and settings have been checked for the selected GPQA slice; label accuracy has not been independently established.

Research question: can a cost-aware router use failure feedback to improve recovery within a limited retry budget when historical estimates become unreliable? A failed response is not necessarily an incorrect routing decision. Repetition must be compared against affordable alternatives under the remaining budget.

## Candidate sources

- [DARS dataset](https://huggingface.co/datasets/AIGNLAI/DARS) and [paper](https://arxiv.org/abs/2606.06924): released scored generations across multiple models and GPQA, MATH-500, and DROP-800. Documentation describes five decoding samples per training rewrite and three decoding samples of each original test question. Keep rewrites separate from same-prompt retries. Paper and dataset metadata must be checked against the actual files.
- [RouterBench](https://github.com/withmartian/routerbench): established routing benchmark with released outcomes and costs. Do not assume it contains repeated stochastic generations for every question/model. One response per pair cannot establish whether retrying the same model helps.
- [PPE](https://github.com/lmarena/PPE): repeated generations exist, but the published curation filters based on correctness and samples questions separately for different models. This can bias retry estimates and limit common-question coverage.
- [BEST-Route](https://github.com/microsoft/best-route-llm): relevant multi-sampling routing work. The checked README describes generating and scoring responses; availability of a complete downloadable scored table has not been established.
- [EvalPlus HumanEval+ sample release](https://github.com/evalplus/evalplus/releases/tag/v0.1.0): stronger candidate for estimating stochastic retry behaviour. The release documents 200 random samples under non-greedy settings and multiple models/temperatures. Its public asset listing includes CodeGen 2B/6B/16B and Code Llama 7B/13B/34B sample archives. HumanEval+ has executable correctness checks; EvalPlus is published at NeurIPS 2023. Archive contents, per-sample labels, prompt/evaluator versions, and common task coverage still need auditing. Keep temperature fixed in the main experiment. Older models, few unique tasks, and unverified cost metadata limit generalization. Test fixes since the initial release mean old labels cannot silently be combined with a newer evaluator. The sample archives are not model weights.

## Recommendation after source review

Correction after file inspection: GPQA's three original-test observations use different temperatures. They DO NOT provide three fixed-configuration retries. Use source-training rewrite 0 with five fixed-setting decodes per model and a new 140/60 parent-question split. The selected GPQA slice has complete coverage: 200 questions × six models × five decodes. Its Qwen/Gemini truncation rates are 70%/80%, so conclusions concern these recorded generation configurations. The published DARS evaluation selects once, so sequential feedback/retry replay is our own protocol.

Investigate EvalPlus as the stronger follow-up for repeat-versus-switch estimates and attempt caps up to five. Its many samples per task address a different weakness from DARS's broader task coverage. Availability of pre-generated code has been confirmed in the official release asset listing; availability and consistency of ready-to-use per-sample correctness labels have not. If evaluation is necessary, execute generated programs only in an appropriate sandbox.

For credible FYP evidence, combine a controlled mechanism diagnostic with empirical replay on at least two tasks or sources. Describe results as offline routing with a specified feedback oracle. They do not establish conversational self-correction, live model drift, or Qwen-specific results. A cost-aware result requires documented cost data or explicitly declared cost proxies plus sensitivity analysis. Replaying a fixed dataset for more warm-up requests does not create additional independent training prompts.

The reason to prefer empirical replay is that task difficulty, cross-model complementarity, and within-model variability come from recorded generations rather than chosen success probabilities. This reduces one source of researcher discretion; it does not remove data selection, scoring, or deployment-validity concerns.

## Audit required before adoption

1. Pin dataset revision; record files, hashes, provenance, and applicable dataset terms.
2. Count unique parent questions, models, exact prompt variants, and independent samples; measure complete cross-model coverage and missing/duplicate records.
3. Check split disjointness by parent question and normalized text. Every rewrite of a question stays in one partition.
4. Verify scoring conventions, parsing failures, truncation, generation settings, and outcome-based filtering. Audit a stratified sample of labels.
5. Check costs: raw versus normalized, per-response versus fixed, and whether normalization uses held-out data. Use a declared cost protocol consistently across policies.
6. Use no more observed same-prompt retries than the files support for the main empirical result. Any resampling beyond that is an explicitly labelled empirical simulation.
7. Keep unselected outcomes hidden from the router. Randomize stored-sample ordering and pair it across policies. Distinguish offline replay from conversational self-correction.

## Initial experiment if the audit passes

Start with the newly split GPQA source-training subset and attempt caps 1, 2, 3, 5. Compare original implemented PromptWise, exponentially weighted PromptWise, repeat-initial, and highest-ranked-untried switching. Preserve the original GSM8K judge; use a separate adapter for the dataset's published labels. Use recorded samples without replacement within an episode, paired across policies.

Report success within budget, total cost including failures, repeat/switch/stop rates, paired uncertainty, and explanatory traces. Separate held-out-domain evaluation from artificial model degradation. Keep implementation-to-paper differences visible, including the local exploration formula and Gram prior. EWMA's separate warm-up checkpoint is not a pure within-request-update intervention.

An FYP-level claim needs robustness across more than one task/source and, when feasible, a small independently collected validation set. Public release and a paper do not by themselves establish label accuracy or retry suitability.
