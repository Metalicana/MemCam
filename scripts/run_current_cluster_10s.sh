#!/bin/bash
set -euo pipefail

GPU="${GPU:-0}"
POLICY="${POLICY:-baseline}"
STEPS="${STEPS:-50}"
MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/ab575577/MemCam/outputs/context_memory/$POLICY}"

cd "$MEMCAM_ROOT"
mkdir -p "$OUTPUT_DIR"

python utils/run_context_memory_batch.py \
  --manifest "$MANIFEST" \
  --gpu "$GPU" \
  --durations 10 \
  --output_dir "$OUTPUT_DIR" \
  --num_inference_steps "$STEPS"
