#!/bin/bash
# Generate MemCam's mce policy at 60s, matching the run_name convention
# (mce_b32 / mce_b64) that evaluate_prefix_duration_curves_60s.sh and
# make_today_fvd_lpips_prefix_plots_60s.sh already expect from RUNS_B32 /
# RUNS_B64 (see the existing baseline/fifo_b*/slam_b*_covisibility/
# ri_b*_dino_rgb runs under $MODEL_ROOT).
#
# Usage:
#   MEMORY_BUDGET=32 bash scripts/run_context_memory_mce_60s.sh
#   MEMORY_BUDGET=64 bash scripts/run_context_memory_mce_60s.sh
set -euo pipefail

GPU="${GPU:-0}"
MEMORY_BUDGET="${MEMORY_BUDGET:?Set MEMORY_BUDGET (e.g. 32 or 64) to match an existing RUNS_B* sweep}"
STEPS="${STEPS:-50}"
MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/memcam_results/context_memory_60s}"
RUN_NAME="${RUN_NAME:-mce_b${MEMORY_BUDGET}}"
OUTPUT_DIR="${OUTPUT_DIR:-$MODEL_ROOT/$RUN_NAME}"

MCE_ALPHA="${MCE_ALPHA:-0.65}"
MCE_LAMBDA="${MCE_LAMBDA:-}"
MCE_GAMMA="${MCE_GAMMA:-0.25}"

cd "$MEMCAM_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "Generating $RUN_NAME (mce, budget $MEMORY_BUDGET) at 60s into $OUTPUT_DIR"

cmd=(
  python utils/run_context_memory_batch.py
  --manifest "$MANIFEST"
  --gpu "$GPU"
  --durations 60
  --output_dir "$OUTPUT_DIR"
  --num_inference_steps "$STEPS"
  --memory_policy mce
  --memory_budget "$MEMORY_BUDGET"
  --mce_alpha "$MCE_ALPHA"
  --mce_gamma "$MCE_GAMMA"
)

if [ -n "$MCE_LAMBDA" ]; then
  cmd+=(--mce_lambda "$MCE_LAMBDA")
fi

"${cmd[@]}"

echo
echo "Done. To fold this into the existing 60s duration-curve comparison:"
echo "  RUNS_B${MEMORY_BUDGET}=\"baseline,fifo_b${MEMORY_BUDGET},slam_b${MEMORY_BUDGET}_covisibility,ri_b${MEMORY_BUDGET}_dino_rgb,mce_b${MEMORY_BUDGET}\" \\"
echo "  bash scripts/make_today_fvd_lpips_prefix_plots_60s.sh"
