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
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/MemCam/outputs/context_memory}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR"
cd "$MEMCAM_ROOT"

python utils/run_context_memory_batch.py \
  --manifest "$MANIFEST" \
  --gpu 0 \
  --rows 0 \
  --output_dir "$OUTPUT_DIR" \
  --smoke
