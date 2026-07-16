#!/bin/bash
#SBATCH --job-name=memcam_cut3r_180s
#SBATCH --output=logs/memcam_cut3r_180s_%A_%a.out
#SBATCH --error=logs/memcam_cut3r_180s_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-16%5

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
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory_180s/manifest.jsonl}"
ROOT="${ROOT:-$HOME/memcam_results/context_180s}"
OUT_DIR="${OUT_DIR:-$ROOT/report_180s/cut3r_pose_recon}"
CUT3R_MODEL="${CUT3R_MODEL:-$MEMCAM_ROOT/CUT3R/src/cut3r_512_dpt_4_64.pth}"

RUNS=(
  baseline
  fifo_b16 fifo_b32 fifo_b64 fifo_b128
  ri_b16_dino_rgb ri_b32_dino_rgb ri_b64_dino_rgb ri_b128_dino_rgb
  slam_b16_covisibility slam_b32_covisibility slam_b64_covisibility slam_b128_covisibility
  kcenter_b16_dino_pose kcenter_b32_dino_pose kcenter_b64_dino_pose kcenter_b128_dino_pose
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN="${RUNS[$TASK_ID]}"
RUN_DIR="$ROOT/$RUN"
STATUS_PATH="$OUT_DIR/status/$RUN.json"

mkdir -p "$MEMCAM_ROOT/logs" "$OUT_DIR/status"
cd "$MEMCAM_ROOT"

VIDEO_COUNT=$(find "$RUN_DIR" -maxdepth 1 -type f -name '*_180s_custom.mp4' | wc -l)
if [ "$VIDEO_COUNT" -ne 15 ]; then
  echo "$RUN has $VIDEO_COUNT/15 videos; refusing an incomplete CUT3R run." >&2
  exit 1
fi

python utils/run_cut3r_context_memory.py \
  --manifest "$MANIFEST" \
  --root "$ROOT" \
  --runs "$RUN" \
  --output_dir "$OUT_DIR" \
  --status_path "$STATUS_PATH" \
  --cut3r_root "$MEMCAM_ROOT/CUT3R" \
  --model_path "$CUT3R_MODEL" \
  --size 512 \
  --device cuda \
  --frame_stride 30 \
  --max_frames 120 \
  --durations 180

echo "CUT3R complete for $RUN"
cat "$STATUS_PATH"
