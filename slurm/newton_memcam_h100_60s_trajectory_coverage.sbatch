#!/bin/bash
#SBATCH --job-name=memcam_60s_traj
#SBATCH --output=logs/memcam_60s_traj_%j.out
#SBATCH --error=logs/memcam_60s_traj_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=18:00:00

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
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/context_memory_60s}"
BUDGET="${BUDGET:-32}"
STEPS="${STEPS:-50}"
GPU="${GPU:-0}"
OUTPUT_DIR="$OUTPUT_ROOT/trajectory_b${BUDGET}_coverage"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR"
cd "$MEMCAM_ROOT"

echo "MemCam 60s trajectory-coverage experiment"
echo "Policy: trajectory_coverage"
echo "Budget: $BUDGET"
echo "Manifest: $MANIFEST"
echo "Output: $OUTPUT_DIR"
echo "Steps: $STEPS"
echo "Started: $(date)"

python utils/run_context_memory_batch.py \
  --manifest "$MANIFEST" \
  --gpu "$GPU" \
  --durations 60 \
  --output_dir "$OUTPUT_DIR" \
  --num_inference_steps "$STEPS" \
  --memory_policy trajectory_coverage \
  --memory_budget "$BUDGET" \
  --memory_bank_device cpu

echo "Finished: $(date)"
echo "Generated videos: $(find "$OUTPUT_DIR" -maxdepth 1 -name '*_60s_custom.mp4' | wc -l)"
echo "Access traces: $(find "$OUTPUT_DIR/access_traces" -name '*.jsonl' 2>/dev/null | wc -l)"
