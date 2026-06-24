#!/bin/bash
#SBATCH --job-name=memcam_cut3r_60s
#SBATCH --output=logs/memcam_cut3r_60s_%j.out
#SBATCH --error=logs/memcam_cut3r_60s_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate memcam

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
ROOT="${ROOT:-$HOME/memcam_results/context_memory_60s}"
RUNS="${RUNS:-baseline,fifo_b32,fifo_b64,ri_b32_dino_rgb,ri_b64_dino_rgb,slam_b16_covisibility,slam_b32_covisibility,slam_b64_covisibility}"
CUT3R_MODEL="${CUT3R_MODEL:-$MEMCAM_ROOT/CUT3R/src/cut3r_512_dpt_4_64.pth}"
CUT3R_SIZE="${CUT3R_SIZE:-512}"
FRAME_STRIDE="${FRAME_STRIDE:-30}"
MAX_FRAMES="${MAX_FRAMES:-120}"
ROWS="${ROWS:-}"
LIMIT="${LIMIT:-}"
OUT_DIR="${OUT_DIR:-$ROOT/cut3r_pose_recon}"
METRICS_DIR="${METRICS_DIR:-$ROOT/cut3r_camera_metrics}"
DATASET_ROOT="${DATASET_ROOT:-}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUT_DIR" "$METRICS_DIR"
cd "$MEMCAM_ROOT"

echo "MemCam CUT3R 60s camera metric"
echo "Manifest: $MANIFEST"
echo "Generated root: $ROOT"
echo "Runs: $RUNS"
echo "CUT3R model: $CUT3R_MODEL"
echo "CUT3R size: $CUT3R_SIZE"
echo "Frame stride: $FRAME_STRIDE"
echo "Max frames: $MAX_FRAMES"
echo "Output dir: $OUT_DIR"
echo "Metrics dir: $METRICS_DIR"
echo "Rows: ${ROWS:-all selected duration rows}"
echo "Limit: ${LIMIT:-none}"
echo "Started: $(date)"

RUN_ARGS=(
  --manifest "$MANIFEST"
  --root "$ROOT"
  --runs "$RUNS"
  --output_dir "$OUT_DIR"
  --cut3r_root "$MEMCAM_ROOT/CUT3R"
  --model_path "$CUT3R_MODEL"
  --size "$CUT3R_SIZE"
  --device cuda
  --frame_stride "$FRAME_STRIDE"
  --max_frames "$MAX_FRAMES"
  --durations 60
)

if [ -n "$ROWS" ]; then
  RUN_ARGS+=(--rows "$ROWS")
fi
if [ -n "$LIMIT" ]; then
  RUN_ARGS+=(--limit "$LIMIT")
fi

python utils/run_cut3r_context_memory.py "${RUN_ARGS[@]}"

EVAL_ARGS=(
  --cut3r_dir "$OUT_DIR"
  --output_dir "$METRICS_DIR"
  --runs "$RUNS"
  --durations 60
)

if [ -n "$ROWS" ]; then
  EVAL_ARGS+=(--rows "$ROWS")
fi
if [ -n "$DATASET_ROOT" ]; then
  EVAL_ARGS+=(--dataset_root "$DATASET_ROOT")
fi

python utils/evaluate_cut3r_camera_metrics.py "${EVAL_ARGS[@]}"

echo "Finished: $(date)"
echo "Summary:"
cat "$METRICS_DIR/cut3r_camera_summary.csv"
