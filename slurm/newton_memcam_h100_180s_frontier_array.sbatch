#!/bin/bash
#SBATCH --job-name=memcam_180s_frontier
#SBATCH --output=logs/memcam_180s_frontier_%A_%a.out
#SBATCH --error=logs/memcam_180s_frontier_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --array=0-12%5

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
TESTBED_DIR="${TESTBED_DIR:-$MEMCAM_ROOT/testbeds/context_memory_180s}"
MANIFEST="${MANIFEST:-$TESTBED_DIR/manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/context_180s}"
DATASET_ROOT="${DATASET_ROOT:-}"
SPLIT_SEED="${SPLIT_SEED:-0}"
SCENES_PER_SPLIT="${SCENES_PER_SPLIT:-15}"
STEPS="${STEPS:-50}"
GPU="${GPU:-0}"
ROWS="${ROWS:-}"

POLICY_NAMES=(
  baseline
  fifo_b16
  fifo_b32
  fifo_b64
  fifo_b128
  ri_b16_dino_rgb
  ri_b32_dino_rgb
  ri_b64_dino_rgb
  ri_b128_dino_rgb
  slam_b16_covisibility
  slam_b32_covisibility
  slam_b64_covisibility
  slam_b128_covisibility
)
MEMORY_POLICIES=(
  unbounded
  fifo
  fifo
  fifo
  fifo
  rarity_irreplaceability
  rarity_irreplaceability
  rarity_irreplaceability
  rarity_irreplaceability
  slam_covisibility
  slam_covisibility
  slam_covisibility
  slam_covisibility
)
MEMORY_BUDGETS=(
  ""
  16
  32
  64
  128
  16
  32
  64
  128
  16
  32
  64
  128
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
POLICY="${POLICY_NAMES[$TASK_ID]}"
MEMORY_POLICY="${MEMORY_POLICIES[$TASK_ID]}"
MEMORY_BUDGET="${MEMORY_BUDGETS[$TASK_ID]}"
OUTPUT_DIR="$OUTPUT_ROOT/$POLICY"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_ROOT" "$OUTPUT_DIR" "$TESTBED_DIR"
cd "$MEMCAM_ROOT"

if [ -z "$DATASET_ROOT" ]; then
  DATASET_ROOT="$(BASE_MANIFEST="$BASE_MANIFEST" python - <<'PY'
import json
import os
from pathlib import Path

manifest = Path(os.environ["BASE_MANIFEST"])
with manifest.open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("gt_frames_dir"):
            print(Path(row["gt_frames_dir"]).expanduser().resolve().parents[1])
            raise SystemExit
raise SystemExit("Could not infer DATASET_ROOT from BASE_MANIFEST.")
PY
)"
fi

echo "MemCam 180s memory frontier"
echo "Task id: $TASK_ID"
echo "Policy: $POLICY"
echo "Memory policy: $MEMORY_POLICY"
echo "Memory budget: ${MEMORY_BUDGET:-none}"
echo "Dataset root: $DATASET_ROOT"
echo "Manifest: $MANIFEST"
echo "Output root: $OUTPUT_ROOT"
echo "Rows: ${ROWS:-all}"
echo "Steps: $STEPS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Started: $(date)"

if [ ! -s "$MANIFEST" ]; then
  LOCKDIR="$TESTBED_DIR/.manifest.lock"
  if mkdir "$LOCKDIR" 2>/dev/null; then
    cleanup_lock() {
      rmdir "$LOCKDIR" 2>/dev/null || true
    }
    trap cleanup_lock EXIT

    echo "Creating 180s manifest..."
    python utils/create_context_memory_testbed.py \
      --dataset_root "$DATASET_ROOT" \
      --output_dir "$TESTBED_DIR" \
      --seeds "$SPLIT_SEED" \
      --scenes_per_split "$SCENES_PER_SPLIT" \
      --durations 180

    cleanup_lock
    trap - EXIT
  else
    echo "Another task is creating the manifest; waiting for $MANIFEST"
    while [ ! -s "$MANIFEST" ]; do
      sleep 10
    done
  fi
fi

python - <<'PY'
import imageio.v2 as imageio
import numpy as np

frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
imageio.mimsave("/tmp/memcam_ffmpeg_180s_frontier_check.mp4", frames, fps=2)
print("imageio ffmpeg writer ok")
PY

cmd=(
  python utils/run_context_memory_batch.py
  --manifest "$MANIFEST"
  --gpu "$GPU"
  --durations 180
  --output_dir "$OUTPUT_DIR"
  --num_inference_steps "$STEPS"
  --memory_policy "$MEMORY_POLICY"
)

if [ -n "$ROWS" ]; then
  cmd+=(--rows "$ROWS")
fi
if [ -n "$MEMORY_BUDGET" ]; then
  cmd+=(--memory_budget "$MEMORY_BUDGET")
fi

"${cmd[@]}"

echo "Finished: $(date)"
echo "== $POLICY =="
echo "videos: $(find "$OUTPUT_DIR" -maxdepth 1 -name '*_180s_custom.mp4' | wc -l)"
echo "traces: $(find "$OUTPUT_DIR/access_traces" -name '*.jsonl' 2>/dev/null | wc -l)"
