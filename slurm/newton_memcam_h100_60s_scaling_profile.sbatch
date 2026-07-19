#!/bin/bash
#SBATCH --job-name=memcam_scale
#SBATCH --output=logs/memcam_scale_%A_%a.out
#SBATCH --error=logs/memcam_scale_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --array=0-1%1

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate memcam

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/memory_latency_scaling_60s}"
STEPS="${STEPS:-50}"
SAMPLES="${SAMPLES:-3}"
BANK_DEVICE="${BANK_DEVICE:-cpu}"
GPU="${GPU:-0}"

RUN_NAMES=(unbounded ri_b32)
MEMORY_POLICIES=(unbounded rarity_irreplaceability)
MEMORY_BUDGETS=("" 32)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_NAME="${RUN_NAMES[$TASK_ID]}_${BANK_DEVICE}"
MEMORY_POLICY="${MEMORY_POLICIES[$TASK_ID]}"
MEMORY_BUDGET="${MEMORY_BUDGETS[$TASK_ID]}"
OUTPUT_DIR="$OUTPUT_ROOT/videos/$RUN_NAME"
PROFILE_DIR="$OUTPUT_ROOT/profiles/$RUN_NAME"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR" "$PROFILE_DIR"
cd "$MEMCAM_ROOT"

echo "Run: $RUN_NAME"
echo "Policy: $MEMORY_POLICY"
echo "Budget: ${MEMORY_BUDGET:-none}"
echo "Bank device: $BANK_DEVICE"
echo "Samples: $SAMPLES"
echo "Profiles: $PROFILE_DIR"

cmd=(
  python utils/run_context_memory_batch.py
  --manifest "$MANIFEST"
  --gpu "$GPU"
  --durations 60
  --max_items "$SAMPLES"
  --output_dir "$OUTPUT_DIR"
  --profile_dir "$PROFILE_DIR"
  --num_inference_steps "$STEPS"
  --memory_policy "$MEMORY_POLICY"
  --memory_bank_device "$BANK_DEVICE"
)

if [ -n "$MEMORY_BUDGET" ]; then
  cmd+=(--memory_budget "$MEMORY_BUDGET")
fi

"${cmd[@]}"

echo "Completed: $(date)"
find "$PROFILE_DIR" -maxdepth 1 -name '*.jsonl' -print
