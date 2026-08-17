#!/bin/bash
#SBATCH --job-name=memcam_slamri_grid
#SBATCH --output=logs/memcam_slamri_grid_%A_%a.out
#SBATCH --error=logs/memcam_slamri_grid_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc33,evc40,evc44
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --array=0-2%1

# Complete the clean k=3 SLAM+RI grid without rerunning existing points:
#   task 0: corrected RI-k3              (beta=0 endpoint)
#   task 1: min-max score blend beta=.25
#   task 2: min-max score blend beta=.75
# Existing runs supply beta=.50 (slamri_b32_beta0p5) and beta=1 (SLAM B32).
# Each task generates the same 15 60-second trajectories and then evaluates
# 10/20/30/60-second LPIPS and FVD. Generation is resume-safe.

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_slamri_grid_mpl_$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/memcam_slamri_grid_xdg_$USER}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/context_memory_60s}"
METRICS_ROOT="${METRICS_ROOT:-$HOME/memcam_results/eval_prefix_duration_curves_60s_b32}"
ROWS="${ROWS:-3,8,13,18,23,28,33,38,43,48,53,58,63,68,73}"
STEPS="${STEPS:-50}"
BUDGET="${BUDGET:-32}"
RI_NEIGHBORS="${RI_NEIGHBORS:-3}"
GPU="${GPU:-0}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RUN_NAME="ri_b${BUDGET}_dino_rgb_k${RI_NEIGHBORS}"
    POLICY="rarity_irreplaceability"
    BETA=""
    ;;
  1)
    RUN_NAME="slamri_b${BUDGET}_beta0p25_k${RI_NEIGHBORS}"
    POLICY="slam_ri_blend"
    BETA="0.25"
    ;;
  2)
    RUN_NAME="slamri_b${BUDGET}_beta0p75_k${RI_NEIGHBORS}"
    POLICY="slam_ri_blend"
    BETA="0.75"
    ;;
  *)
    echo "Unsupported array task: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR" "$OUTPUT_DIR/profiles" "$METRICS_ROOT"
cd "$MEMCAM_ROOT"

echo "MemCam clean RI-k3 / SLAM+RI grid"
echo "Task: ${SLURM_ARRAY_TASK_ID}"
echo "Run: $RUN_NAME"
echo "Policy: $POLICY"
echo "Beta: ${BETA:-RI endpoint}"
echo "RI rarity neighbors: $RI_NEIGHBORS"
echo "Budget: $BUDGET"
echo "Rows: $ROWS"
echo "Manifest: $MANIFEST"
echo "Output: $OUTPUT_DIR"
echo "Git commit: $(git rev-parse HEAD)"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

timeout 90s python -c \
  'import torch; assert torch.cuda.is_available(); print("CUDA:", torch.cuda.get_device_name(0))'

generation_cmd=(
  python utils/run_context_memory_batch.py
  --manifest "$MANIFEST"
  --gpu "$GPU"
  --durations 60
  --rows "$ROWS"
  --output_dir "$OUTPUT_DIR"
  --num_inference_steps "$STEPS"
  --memory_policy "$POLICY"
  --memory_budget "$BUDGET"
  --memory_bank_device cpu
  --profile_dir "$OUTPUT_DIR/profiles"
)

if [ "$POLICY" = "rarity_irreplaceability" ]; then
  generation_cmd+=(--ri_rarity_neighbors "$RI_NEIGHBORS")
else
  generation_cmd+=(
    --slamri_beta "$BETA"
    --slamri_rarity_neighbors "$RI_NEIGHBORS"
  )
fi

printf 'Generation command:'
printf ' %q' "${generation_cmd[@]}"
printf '\n'
"${generation_cmd[@]}"

video_count=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*_60s_custom.mp4' | wc -l)
echo "Generated videos: $video_count/15"
if [ "$video_count" -ne 15 ]; then
  echo "Expected 15 videos before evaluation, found $video_count" >&2
  exit 1
fi

python utils/evaluate_context_memory_prefix_curves.py \
  --manifest "$MANIFEST" \
  --model_output_dir "$OUTPUT_DIR" \
  --metrics_dir "$METRICS_ROOT" \
  --run_name "$RUN_NAME" \
  --rows "$ROWS" \
  --source_duration 60 \
  --eval_durations 10,20,30,60 \
  --learned_metrics lpips,fvd \
  --metric_device cuda \
  --metric_batch_size 8 \
  --learned_image_size 224 \
  --frame_stride 30 \
  --fvd_clip_length 16 \
  --fvd_clips_per_video 4 \
  --fvd_frame_stride 4 \
  --fvd_image_size 224 \
  --fvd_backend styleganv_i3d \
  --fvd_cache_dir "$HF_HOME/memcam_fvd" \
  --strict

echo "Finished: $(date)"
echo "Summary: $METRICS_ROOT/$RUN_NAME/summary.json"
cat "$METRICS_ROOT/$RUN_NAME/summary.json"
