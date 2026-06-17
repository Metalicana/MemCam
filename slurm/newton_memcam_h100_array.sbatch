#!/bin/bash
#SBATCH --job-name=memcam_h100
#SBATCH --output=logs/memcam_h100_%A_%a.out
#SBATCH --error=logs/memcam_h100_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-74%8

set -euo pipefail

module load anaconda
conda activate memcam

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/scratch/$USER/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
POLICY="${POLICY:-baseline}"
MEMORY_POLICY="${MEMORY_POLICY:-$POLICY}"
if [ "$MEMORY_POLICY" = "baseline" ]; then
  MEMORY_POLICY="unbounded"
fi
MEMORY_BUDGET="${MEMORY_BUDGET:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/MemCam/outputs/context_memory/$POLICY}"
STEPS="${STEPS:-50}"

if { [ "$MEMORY_POLICY" = "fifo" ] || [ "$MEMORY_POLICY" = "rarity_irreplaceability" ]; } && [ -z "$MEMORY_BUDGET" ]; then
  echo "MEMORY_BUDGET is required when MEMORY_POLICY=$MEMORY_POLICY" >&2
  exit 2
fi

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR"
cd "$MEMCAM_ROOT"

cmd=(
  python utils/run_context_memory_batch.py
  --manifest "$MANIFEST"
  --gpu 0
  --rows "$SLURM_ARRAY_TASK_ID"
  --output_dir "$OUTPUT_DIR"
  --num_inference_steps "$STEPS"
  --memory_policy "$MEMORY_POLICY"
)

if [ -n "$MEMORY_BUDGET" ]; then
  cmd+=(--memory_budget "$MEMORY_BUDGET")
fi

"${cmd[@]}"
