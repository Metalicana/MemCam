#!/bin/bash
#SBATCH --job-name=memcam_60s_plus15
#SBATCH --output=logs/memcam_60s_plus15_%A_%a.out
#SBATCH --error=logs/memcam_60s_plus15_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --array=0-6%4

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
BASE_MANIFEST="${BASE_MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
MANIFEST_30="${MANIFEST_30:-$MEMCAM_ROOT/testbeds/context_memory/manifest_60s_30.jsonl}"
NEW_ROWS_PATH="${NEW_ROWS_PATH:-$MEMCAM_ROOT/testbeds/context_memory/manifest_60s_30.new_rows.txt}"
MANIFEST_METADATA="${MANIFEST_METADATA:-$MEMCAM_ROOT/testbeds/context_memory/manifest_60s_30.metadata.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/context_memory_60s}"
TARGET_COUNT="${TARGET_COUNT:-30}"
EXTRA_SEED="${EXTRA_SEED:-1}"
STEPS="${STEPS:-50}"
GPU="${GPU:-0}"
ROWS="${ROWS:-}"

POLICY_NAMES=(
  baseline
  fifo_b32
  fifo_b64
  ri_b32_dino_rgb
  ri_b64_dino_rgb
  slam_b32_covisibility
  slam_b64_covisibility
)
MEMORY_POLICIES=(
  unbounded
  fifo
  fifo
  rarity_irreplaceability
  rarity_irreplaceability
  slam_covisibility
  slam_covisibility
)
MEMORY_BUDGETS=(
  ""
  32
  64
  32
  64
  32
  64
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
POLICY="${POLICY_NAMES[$TASK_ID]}"
MEMORY_POLICY="${MEMORY_POLICIES[$TASK_ID]}"
MEMORY_BUDGET="${MEMORY_BUDGETS[$TASK_ID]}"
OUTPUT_DIR="$OUTPUT_ROOT/$POLICY"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_ROOT" "$OUTPUT_DIR"
cd "$MEMCAM_ROOT"

echo "MemCam 60s plus-15 array"
echo "Task id: $TASK_ID"
echo "Policy: $POLICY"
echo "Memory policy: $MEMORY_POLICY"
echo "Memory budget: ${MEMORY_BUDGET:-none}"
echo "Base manifest: $BASE_MANIFEST"
echo "30-video manifest: $MANIFEST_30"
echo "New rows path: $NEW_ROWS_PATH"
echo "Output root: $OUTPUT_ROOT"
echo "Steps: $STEPS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Started: $(date)"

python utils/extend_context_memory_60s_manifest.py \
  --input_manifest "$BASE_MANIFEST" \
  --output_manifest "$MANIFEST_30" \
  --target_count "$TARGET_COUNT" \
  --duration 60 \
  --seed "$EXTRA_SEED" \
  --new_rows_path "$NEW_ROWS_PATH" \
  --metadata_path "$MANIFEST_METADATA"

if [ -z "$ROWS" ]; then
  ROWS="$(cat "$NEW_ROWS_PATH")"
fi

if [ -z "$ROWS" ]; then
  echo "No new rows requested; exiting."
  exit 0
fi

echo "Rows to run: $ROWS"

python - <<'PY'
import imageio.v2 as imageio
import numpy as np

frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
imageio.mimsave("/tmp/memcam_ffmpeg_slurm_check.mp4", frames, fps=2)
print("imageio ffmpeg writer ok")
PY

cmd=(
  python utils/run_context_memory_batch.py
  --manifest "$MANIFEST_30"
  --gpu "$GPU"
  --rows "$ROWS"
  --durations 60
  --output_dir "$OUTPUT_DIR"
  --num_inference_steps "$STEPS"
  --memory_policy "$MEMORY_POLICY"
)

if [ -n "$MEMORY_BUDGET" ]; then
  cmd+=(--memory_budget "$MEMORY_BUDGET")
fi

"${cmd[@]}"

echo "Finished: $(date)"
echo "== $POLICY =="
echo "videos: $(find "$OUTPUT_DIR" -maxdepth 1 -name '*_60s_custom.mp4' | wc -l)"
echo "traces: $(find "$OUTPUT_DIR/access_traces" -name '*.jsonl' 2>/dev/null | wc -l)"
