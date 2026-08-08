#!/bin/bash
#SBATCH --job-name=memcam_60s_fill15
#SBATCH --output=logs/memcam_60s_fill15_%A_%a.out
#SBATCH --error=logs/memcam_60s_fill15_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-11%3

# Brings every (policy, budget) combo in the B in {16,32,64,128} sweep up to
# the standard 15-row pilot set (rows 3,8,13,18,23,28,33,38,43,48,53,58,63,
# 68,73), skipping whatever's already done per
# `utils/audit_context_memory_runs.py`. unbounded/baseline is excluded --
# already at 15/15 and budget-independent. Rerun the audit after this
# finishes to confirm everything landed.

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate memcam

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
mkdir -p "$HF_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/context_memory_60s}"
STEPS="${STEPS:-50}"
GPU="${GPU:-0}"

STANDARD_15="3,8,13,18,23,28,33,38,43,48,53,58,63,68,73"

# Parallel arrays, one entry per task.
POLICY_NAMES=(
  fifo_b16
  slam_b128_covisibility
  ri_b16_dino_rgb
  ri_b128_dino_rgb
  kcenter_b16
  kcenter_b32
  kcenter_b64
  kcenter_b128
  mce_b16_lambda1_pilot
  mce_b32_lambda1_pilot
  mce_b64_lambda1_pilot
  mce_b128_lambda1_pilot
)
MEMORY_POLICIES=(
  fifo
  slam_covisibility
  rarity_irreplaceability
  rarity_irreplaceability
  kcenter_coreset
  kcenter_coreset
  kcenter_coreset
  kcenter_coreset
  mce
  mce
  mce
  mce
)
MEMORY_BUDGETS=(
  16
  128
  16
  128
  16
  32
  64
  128
  16
  32
  64
  128
)
# Empty string = full standard 15; anything else overrides to just the gap.
ROWS_OVERRIDE=(
  ""
  "18,23,28,33,38,43,48,53,58,63,68,73"
  ""
  "63,68,73"
  ""
  ""
  ""
  ""
  ""
  "8,13,18,23,33,38,43,48,58,63,68,73"
  ""
  ""
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
POLICY="${POLICY_NAMES[$TASK_ID]}"
MEMORY_POLICY="${MEMORY_POLICIES[$TASK_ID]}"
MEMORY_BUDGET="${MEMORY_BUDGETS[$TASK_ID]}"
ROWS="${ROWS_OVERRIDE[$TASK_ID]:-$STANDARD_15}"
OUTPUT_DIR="$OUTPUT_ROOT/$POLICY"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_ROOT" "$OUTPUT_DIR" "$OUTPUT_DIR/profiles"
cd "$MEMCAM_ROOT"

echo "MemCam 60s fill-to-15 array"
echo "Task id: $TASK_ID"
echo "Output dir: $POLICY"
echo "Memory policy: $MEMORY_POLICY"
echo "Memory budget: $MEMORY_BUDGET"
echo "Rows: $ROWS"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

python utils/check_memory_policies.py

cmd=(
  python utils/run_context_memory_batch.py
  --manifest "$MANIFEST"
  --gpu "$GPU"
  --rows "$ROWS"
  --durations 60
  --output_dir "$OUTPUT_DIR"
  --num_inference_steps "$STEPS"
  --memory_policy "$MEMORY_POLICY"
  --memory_budget "$MEMORY_BUDGET"
  --profile_dir "$OUTPUT_DIR/profiles"
)
if [ "$MEMORY_POLICY" = "mce" ]; then
  cmd+=(--mce_lambda 1.0 --memory_bank_device cpu)
fi

"${cmd[@]}"

echo "Finished: $(date)"
echo "== $POLICY =="
echo "videos: $(find "$OUTPUT_DIR" -maxdepth 1 -name '*_60s_custom.mp4' | wc -l)"
echo "traces: $(find "$OUTPUT_DIR/access_traces" -name '*.jsonl' 2>/dev/null | wc -l)"
