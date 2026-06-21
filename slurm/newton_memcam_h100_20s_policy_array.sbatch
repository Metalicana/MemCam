#!/bin/bash
#SBATCH --job-name=memcam_20s_policy
#SBATCH --output=logs/memcam_20s_policy_%A_%a.out
#SBATCH --error=logs/memcam_20s_policy_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-14%5

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
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
POLICY="${POLICY:-baseline}"
MEMORY_POLICY="${MEMORY_POLICY:-$POLICY}"
if [ "$MEMORY_POLICY" = "baseline" ]; then
  MEMORY_POLICY="unbounded"
fi
MEMORY_BUDGET="${MEMORY_BUDGET:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/context_memory_20s}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$POLICY}"
STEPS="${STEPS:-50}"

if { [ "$MEMORY_POLICY" = "fifo" ] || [ "$MEMORY_POLICY" = "rarity_irreplaceability" ] || [ "$MEMORY_POLICY" = "slam_covisibility" ]; } && [ -z "$MEMORY_BUDGET" ]; then
  echo "MEMORY_BUDGET is required when MEMORY_POLICY=$MEMORY_POLICY" >&2
  exit 2
fi

# Manifest order is scene-major with durations [10, 20, 40, 60, 120].
# Therefore 20s rows are 1, 6, 11, ..., 71.
ROW=$((SLURM_ARRAY_TASK_ID * 5 + 1))

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR"
cd "$MEMCAM_ROOT"

echo "Policy: $POLICY"
echo "Memory policy: $MEMORY_POLICY"
echo "Memory budget: ${MEMORY_BUDGET:-none}"
echo "Manifest row: $ROW"
echo "Output dir: $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

cmd=(
  python utils/run_context_memory_batch.py
  --manifest "$MANIFEST"
  --gpu 0
  --rows "$ROW"
  --output_dir "$OUTPUT_DIR"
  --num_inference_steps "$STEPS"
  --memory_policy "$MEMORY_POLICY"
)

if [ -n "$MEMORY_BUDGET" ]; then
  cmd+=(--memory_budget "$MEMORY_BUDGET")
fi

"${cmd[@]}"
