#!/bin/bash
#SBATCH --job-name=memcam_eval_180s
#SBATCH --output=logs/memcam_eval_180s_%A_%a.out
#SBATCH --error=logs/memcam_eval_180s_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_mpl_$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/memcam_xdg_$USER}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory_180s/manifest.jsonl}"
ROOT="${ROOT:-$HOME/memcam_results/context_180s}"
METRICS_DIR="${METRICS_DIR:-$ROOT/report_180s/eval_metrics}"

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

mkdir -p "$MEMCAM_ROOT/logs" "$METRICS_DIR"
cd "$MEMCAM_ROOT"

VIDEO_COUNT=$(find "$RUN_DIR" -maxdepth 1 -type f -name '*_180s_custom.mp4' | wc -l)
if [ "$VIDEO_COUNT" -ne 15 ]; then
  echo "$RUN has $VIDEO_COUNT/15 videos; refusing an incomplete evaluation." >&2
  exit 1
fi

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
PY

echo "Evaluating $RUN on all 15 180s videos"
python utils/evaluate_context_memory_prefix_curves.py \
  --manifest "$MANIFEST" \
  --model_output_dir "$RUN_DIR" \
  --metrics_dir "$METRICS_DIR" \
  --run_name "$RUN" \
  --source_duration 180 \
  --eval_durations 180 \
  --frame_stride 90 \
  --learned_metrics lpips,dino,fvd \
  --metric_device cuda \
  --metric_batch_size 8 \
  --learned_image_size 224 \
  --fvd_clip_length 16 \
  --fvd_clips_per_video 4 \
  --fvd_frame_stride 4 \
  --fvd_image_size 224 \
  --fvd_backend styleganv_i3d \
  --fvd_cache_dir "$HF_HOME/memcam_fvd" \
  --write_frame_metrics \
  --strict

echo "Summary: $METRICS_DIR/$RUN/summary.json"
cat "$METRICS_DIR/$RUN/summary.json"
