#!/bin/bash
#SBATCH --job-name=memcam_smoke
#SBATCH --output=logs/memcam_smoke_%j.out
#SBATCH --error=logs/memcam_smoke_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

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

if [ "$MEMORY_POLICY" = "fifo" ] && [ -z "$MEMORY_BUDGET" ]; then
  echo "MEMORY_BUDGET is required when MEMORY_POLICY=fifo" >&2
  exit 2
fi

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR"
cd "$MEMCAM_ROOT"

cmd=(
  python utils/run_context_memory_batch.py
  --manifest "$MANIFEST"
  --gpu 0
  --rows 0
  --output_dir "$OUTPUT_DIR"
  --smoke
  --memory_policy "$MEMORY_POLICY"
)

if [ -n "$MEMORY_BUDGET" ]; then
  cmd+=(--memory_budget "$MEMORY_BUDGET")
fi

"${cmd[@]}"
