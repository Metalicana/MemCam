#!/bin/bash
set -euo pipefail

GPU="${GPU:-0}"
POLICY="${POLICY:-baseline}"
MEMORY_POLICY="${MEMORY_POLICY:-$POLICY}"
if [ "$MEMORY_POLICY" = "baseline" ]; then
  MEMORY_POLICY="unbounded"
fi
MEMORY_BUDGET="${MEMORY_BUDGET:-}"
STEPS="${STEPS:-50}"
MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/ab575577/MemCam/outputs/context_memory/$POLICY}"

if { [ "$MEMORY_POLICY" = "fifo" ] || [ "$MEMORY_POLICY" = "rarity_irreplaceability" ] || [ "$MEMORY_POLICY" = "slam_covisibility" ] || [ "$MEMORY_POLICY" = "facility_coreset" ]; } && [ -z "$MEMORY_BUDGET" ]; then
  echo "MEMORY_BUDGET is required when MEMORY_POLICY=$MEMORY_POLICY" >&2
  exit 2
fi

cd "$MEMCAM_ROOT"
mkdir -p "$OUTPUT_DIR"

cmd=(
  python utils/run_context_memory_batch.py
  --manifest "$MANIFEST"
  --gpu "$GPU"
  --durations 10
  --output_dir "$OUTPUT_DIR"
  --num_inference_steps "$STEPS"
  --memory_policy "$MEMORY_POLICY"
)

if [ -n "$MEMORY_BUDGET" ]; then
  cmd+=(--memory_budget "$MEMORY_BUDGET")
fi

"${cmd[@]}"
