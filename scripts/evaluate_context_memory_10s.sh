#!/bin/bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-baseline}"
MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-/data/ab575577/MemCam/outputs/context_memory/$RUN_NAME}"
METRICS_DIR="${METRICS_DIR:-/data/ab575577/MemCam/eval/context_memory}"
DATASET_ROOT="${DATASET_ROOT:-}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"

cd "$MEMCAM_ROOT"

cmd=(
  python utils/evaluate_context_memory.py
  --manifest "$MANIFEST"
  --model_output_dir "$MODEL_OUTPUT_DIR"
  --metrics_dir "$METRICS_DIR"
  --run_name "$RUN_NAME"
  --durations 10
  --frame_stride "$FRAME_STRIDE"
)

if [ -n "$DATASET_ROOT" ]; then
  cmd+=(--dataset_root "$DATASET_ROOT")
fi

"${cmd[@]}"
