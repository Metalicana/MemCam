#!/bin/bash
#SBATCH --job-name=memcam_180s_finish
#SBATCH --output=logs/memcam_180s_finish_%A_%a.out
#SBATCH --error=logs/memcam_180s_finish_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-59%8

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate memcam

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
mkdir -p "$HF_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory_180s/manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/context_180s}"
STEPS="${STEPS:-50}"

POLICY_NAMES=(
  baseline
  fifo_b64
  ri_b64_dino_rgb
  ri_b128_dino_rgb
)
MEMORY_POLICIES=(
  unbounded
  fifo
  rarity_irreplaceability
  rarity_irreplaceability
)
MEMORY_BUDGETS=(
  ""
  64
  64
  128
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
POLICY_INDEX=$((TASK_ID / 15))
ROW=$((TASK_ID % 15))
POLICY="${POLICY_NAMES[$POLICY_INDEX]}"
MEMORY_POLICY="${MEMORY_POLICIES[$POLICY_INDEX]}"
MEMORY_BUDGET="${MEMORY_BUDGETS[$POLICY_INDEX]}"
OUTPUT_DIR="$OUTPUT_ROOT/$POLICY"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR"
cd "$MEMCAM_ROOT"

if [ ! -s "$MANIFEST" ]; then
  echo "Missing manifest: $MANIFEST" >&2
  exit 1
fi

MANIFEST_ROWS=$(grep -cve '^[[:space:]]*$' "$MANIFEST")
if [ "$MANIFEST_ROWS" -ne 15 ]; then
  echo "Expected 15 rows in $MANIFEST, found $MANIFEST_ROWS" >&2
  exit 1
fi

echo "MemCam 180s missing-video recovery"
echo "Task: $TASK_ID"
echo "Policy: $POLICY"
echo "Manifest row: $ROW"
echo "Memory policy: $MEMORY_POLICY"
echo "Memory budget: ${MEMORY_BUDGET:-none}"
echo "Started: $(date)"

cmd=(
  python utils/run_context_memory_batch.py
  --manifest "$MANIFEST"
  --durations 180
  --rows "$ROW"
  --output_dir "$OUTPUT_DIR"
  --num_inference_steps "$STEPS"
  --memory_policy "$MEMORY_POLICY"
)

if [ -n "$MEMORY_BUDGET" ]; then
  cmd+=(--memory_budget "$MEMORY_BUDGET")
fi

"${cmd[@]}"

COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*_180s_custom.mp4' | wc -l)
echo "Finished: $(date)"
echo "$POLICY videos: $COUNT/15"
