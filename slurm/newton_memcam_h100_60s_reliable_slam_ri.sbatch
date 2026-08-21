#!/bin/bash
#SBATCH --job-name=memcam_60s_rsri
#SBATCH --output=logs/memcam_60s_rsri_%j.out
#SBATCH --error=logs/memcam_60s_rsri_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc23,evc33,evc40,evc42,evc44
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=36:00:00

# Final policy: reject unreliable supported revisits, then rank every admitted
# frame by 0.75 * normalized SLAM + 0.25 * normalized RI.

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_rsri_mpl_$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/memcam_rsri_xdg_$USER}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/context_memory_60s}"
METRICS_ROOT="${METRICS_ROOT:-$HOME/memcam_results/eval_prefix_duration_curves_60s_b32}"
ROWS="${ROWS:-3,8,13,18,23,28,33,38,43,48,53,58,63,68,73}"
RUN_NAME="${RUN_NAME:-rsri_b32_s75_r25}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
STEPS="${STEPS:-50}"
BUDGET="${BUDGET:-32}"

RSRI_SLAM_WEIGHT="${RSRI_SLAM_WEIGHT:-0.75}"
RSRI_RARITY_NEIGHBORS="${RSRI_RARITY_NEIGHBORS:-3}"
RSRI_RELIABILITY_NEIGHBORS="${RSRI_RELIABILITY_NEIGHBORS:-3}"
RSRI_RELIABILITY_MIN_SUPPORT="${RSRI_RELIABILITY_MIN_SUPPORT:-2}"
RSRI_RELIABILITY_GEOMETRY_THRESHOLD="${RSRI_RELIABILITY_GEOMETRY_THRESHOLD:-0.50}"
RSRI_RELIABILITY_THRESHOLD="${RSRI_RELIABILITY_THRESHOLD:-0.80}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR" "$OUTPUT_DIR/profiles" "$METRICS_ROOT"
cd "$MEMCAM_ROOT"

echo "MemCam 60s Reliable SLAM+RI"
echo "Run: $RUN_NAME"
echo "Budget: $BUDGET"
echo "Rows: $ROWS"
echo "SLAM weight: $RSRI_SLAM_WEIGHT"
echo "RI weight: 1 - $RSRI_SLAM_WEIGHT"
echo "RI neighbors: $RSRI_RARITY_NEIGHBORS"
echo "Reliability references: $RSRI_RELIABILITY_NEIGHBORS"
echo "Reliability minimum support: $RSRI_RELIABILITY_MIN_SUPPORT"
echo "Reliability geometry threshold: $RSRI_RELIABILITY_GEOMETRY_THRESHOLD"
echo "Reliability threshold: $RSRI_RELIABILITY_THRESHOLD"
echo "Git commit: $(git rev-parse HEAD)"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

cuda_ready=0
for attempt in 1 2 3; do
  echo "CUDA preflight attempt $attempt/3"
  if timeout 90s python -c \
    'import torch; assert torch.cuda.is_available(); print("CUDA:", torch.cuda.get_device_name(0))'; then
    cuda_ready=1
    break
  fi
  sleep 15
done
if [ "$cuda_ready" -ne 1 ]; then
  echo "CUDA failed all three preflight attempts on ${SLURMD_NODENAME:-unknown}" >&2
  exit 124
fi

python -m unittest discover -s tests -p 'test_reliable_slam_ri.py' -v
python utils/check_memory_policies.py

python utils/run_context_memory_batch.py \
  --manifest "$MANIFEST" \
  --gpu 0 \
  --durations 60 \
  --rows "$ROWS" \
  --output_dir "$OUTPUT_DIR" \
  --num_inference_steps "$STEPS" \
  --memory_policy reliable_slam_ri \
  --memory_budget "$BUDGET" \
  --memory_bank_device cpu \
  --rsri_slam_weight "$RSRI_SLAM_WEIGHT" \
  --rsri_rarity_neighbors "$RSRI_RARITY_NEIGHBORS" \
  --rsri_reliability_neighbors "$RSRI_RELIABILITY_NEIGHBORS" \
  --rsri_reliability_min_support "$RSRI_RELIABILITY_MIN_SUPPORT" \
  --rsri_reliability_geometry_threshold "$RSRI_RELIABILITY_GEOMETRY_THRESHOLD" \
  --rsri_reliability_threshold "$RSRI_RELIABILITY_THRESHOLD" \
  --profile_dir "$OUTPUT_DIR/profiles"

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
echo "Videos: $OUTPUT_DIR"
echo "Summary: $METRICS_ROOT/$RUN_NAME/summary.json"
cat "$METRICS_ROOT/$RUN_NAME/summary.json"
