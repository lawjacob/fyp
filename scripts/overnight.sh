#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
RUN_DIR="${RUN_DIR:-runs/qwen-retry-v1}"
PYTHON="${PYTHON:-python}"
mkdir -p "$RUN_DIR"
"$PYTHON" -m pip freeze > "$RUN_DIR/environment.txt"
"$PYTHON" -m benchmarks.collect_qwen prepare --output "$RUN_DIR/prompts.jsonl" --train "${TRAIN_PROMPTS:-300}" --eval "${EVAL_PROMPTS:-100}"
"$PYTHON" -m benchmarks.collect_qwen collect --prompts "$RUN_DIR/prompts.jsonl" --output "$RUN_DIR/collection" --samples "${SAMPLES:-5}" --max-new-tokens "${MAX_NEW_TOKENS:-512}"
"$PYTHON" -m benchmarks.collect_qwen pack --prompts "$RUN_DIR/prompts.jsonl" --collection "$RUN_DIR/collection" --output "$RUN_DIR/outcomes.npz" --costs 1 3 6 --cost-note 'Parameter-size relative per-call proxy 1:3:6; not measured latency or dollars'
"$PYTHON" -m benchmarks.retry_benchmark --data "$RUN_DIR/outcomes.npz" --output "$RUN_DIR/replay" --eval-requests "${EVAL_PROMPTS:-100}"
printf 'Finished. Summary: %s/replay/summary.csv\n' "$RUN_DIR"
