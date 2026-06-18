#!/bin/bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-baseline}"
MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-/data/ab575577/MemCam/outputs/context_memory/$RUN_NAME}"
METRICS_DIR="${METRICS_DIR:-/data/ab575577/MemCam/eval/context_memory}"
DATASET_ROOT="${DATASET_ROOT:-}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
LEARNED_METRICS="${LEARNED_METRICS:-none}"
METRIC_DEVICE="${METRIC_DEVICE:-cuda}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-8}"
LEARNED_IMAGE_SIZE="${LEARNED_IMAGE_SIZE:-224}"

cd "$MEMCAM_ROOT"

cmd=(
  python utils/evaluate_context_memory.py
  --manifest "$MANIFEST"
  --model_output_dir "$MODEL_OUTPUT_DIR"
  --metrics_dir "$METRICS_DIR"
  --run_name "$RUN_NAME"
  --durations 10
  --frame_stride "$FRAME_STRIDE"
  --learned_metrics "$LEARNED_METRICS"
  --metric_device "$METRIC_DEVICE"
  --metric_batch_size "$METRIC_BATCH_SIZE"
  --learned_image_size "$LEARNED_IMAGE_SIZE"
)

if [ -n "$DATASET_ROOT" ]; then
  cmd+=(--dataset_root "$DATASET_ROOT")
fi

"${cmd[@]}"
