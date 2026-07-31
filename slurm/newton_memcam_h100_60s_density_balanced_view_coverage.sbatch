#!/bin/bash
#SBATCH --job-name=memcam_60s_dbvc
#SBATCH --output=logs/memcam_60s_dbvc_%j.out
#SBATCH --error=logs/memcam_60s_dbvc_%j.err
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
export PYTHONUNBUFFERED=1
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
DENSITY_ALPHA="${DENSITY_ALPHA:-0.5}"
DINO_WEIGHT="${DINO_WEIGHT:-0.5}"
RGB_WEIGHT="${RGB_WEIGHT:-0.25}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/density_b${BUDGET}_view_coverage}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR" "$OUTPUT_DIR/profiles"
cd "$MEMCAM_ROOT"

echo "MemCam 60s density-balanced view-coverage experiment"
echo "Policy: density_balanced_view_coverage"
echo "Budget: $BUDGET"
echo "Density alpha: $DENSITY_ALPHA"
echo "DINO weight: $DINO_WEIGHT"
echo "RGB weight: $RGB_WEIGHT"
echo "Manifest: $MANIFEST"
echo "Output: $OUTPUT_DIR"
echo "Steps: $STEPS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

timeout 90s python -c \
  'import torch; assert torch.cuda.is_available(); print("CUDA:", torch.cuda.get_device_name(0))'
python utils/check_memory_policies.py

python utils/run_context_memory_batch.py \
  --manifest "$MANIFEST" \
  --gpu "$GPU" \
  --durations 60 \
  --output_dir "$OUTPUT_DIR" \
  --num_inference_steps "$STEPS" \
  --memory_policy density_balanced_view_coverage \
  --memory_budget "$BUDGET" \
  --memory_bank_device cpu \
  --density_coverage_alpha "$DENSITY_ALPHA" \
  --density_coverage_dino_weight "$DINO_WEIGHT" \
  --density_coverage_rgb_weight "$RGB_WEIGHT" \
  --profile_dir "$OUTPUT_DIR/profiles"

echo "Finished: $(date)"
echo "Generated videos: $(find "$OUTPUT_DIR" -maxdepth 1 -name '*_60s_custom.mp4' | wc -l)"
echo "Access traces: $(find "$OUTPUT_DIR/access_traces" -name '*.jsonl' 2>/dev/null | wc -l)"
echo "Profiles: $(find "$OUTPUT_DIR/profiles" -name '*.jsonl' 2>/dev/null | wc -l)"
