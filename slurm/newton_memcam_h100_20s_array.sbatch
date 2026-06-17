#!/bin/bash
#SBATCH --job-name=memcam_20s
#SBATCH --output=logs/memcam_20s_%A_%a.out
#SBATCH --error=logs/memcam_20s_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-14%8

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
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/MemCam/outputs/context_memory/$POLICY}"
STEPS="${STEPS:-50}"

# Manifest order is scene-major with durations [10, 20, 40, 60, 120].
# Therefore 20s rows are 1, 6, 11, ..., 71.
ROW=$((SLURM_ARRAY_TASK_ID * 5 + 1))

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR"
cd "$MEMCAM_ROOT"

python utils/run_context_memory_batch.py \
  --manifest "$MANIFEST" \
  --gpu 0 \
  --rows "$ROW" \
  --output_dir "$OUTPUT_DIR" \
  --num_inference_steps "$STEPS"
