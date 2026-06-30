#!/bin/bash
#SBATCH --job-name=memcam_ri64_missing15
#SBATCH --output=logs/memcam_ri64_missing15_%j.out
#SBATCH --error=logs/memcam_ri64_missing15_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00

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
OUTPUT_DIR="$OUTPUT_ROOT/ri_b64_dino_rgb"
TARGET_COUNT="${TARGET_COUNT:-30}"
EXTRA_SEED="${EXTRA_SEED:-1}"
ROWS="${ROWS:-15-29}"
STEPS="${STEPS:-50}"
GPU="${GPU:-0}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_ROOT" "$OUTPUT_DIR"
cd "$MEMCAM_ROOT"

echo "MemCam RI b64 missing-15 rerun"
echo "Manifest: $MANIFEST_30"
echo "Rows: $ROWS"
echo "Output dir: $OUTPUT_DIR"
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

python - <<'PY'
import imageio.v2 as imageio
import numpy as np

frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
imageio.mimsave("/tmp/memcam_ffmpeg_ri64_missing15_check.mp4", frames, fps=2)
print("imageio ffmpeg writer ok")
PY

python utils/run_context_memory_batch.py \
  --manifest "$MANIFEST_30" \
  --gpu "$GPU" \
  --rows "$ROWS" \
  --durations 60 \
  --output_dir "$OUTPUT_DIR" \
  --num_inference_steps "$STEPS" \
  --memory_policy rarity_irreplaceability \
  --memory_budget 64

echo "Finished: $(date)"
echo "== ri_b64_dino_rgb =="
echo "videos: $(find "$OUTPUT_DIR" -maxdepth 1 -name '*_60s_custom.mp4' | wc -l)"
echo "new seed1 videos: $(find "$OUTPUT_DIR" -maxdepth 1 -name 'seed1_extra60s_*_60s_custom.mp4' | wc -l)"
echo "traces: $(find "$OUTPUT_DIR/access_traces" -name '*.jsonl' 2>/dev/null | wc -l)"
