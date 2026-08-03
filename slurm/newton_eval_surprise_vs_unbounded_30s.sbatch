#!/bin/bash
#SBATCH --job-name=memcam_eval30
#SBATCH --output=logs/memcam_eval30_%j.out
#SBATCH --error=logs/memcam_eval30_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=evc33,evc40,evc44

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate memcam

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_matplotlib_cache_$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/memcam_xdg_cache_$USER}"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
SURPRISE_MANIFEST="${SURPRISE_MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory_surprise_30s_10/manifest.jsonl}"
ATTENTION_MANIFEST="${ATTENTION_MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory_attention_30s_10/manifest.jsonl}"
SURPRISE_VIDEOS="${SURPRISE_VIDEOS:-$HOME/memcam_results/context_memory_surprise_30s/surprise_b32_value}"
UNBOUNDED_VIDEOS="${UNBOUNDED_VIDEOS:-$HOME/memcam_results/context_memory_attention_30s/unbounded_attention_audit}"
METRICS_DIR="${METRICS_DIR:-$HOME/memcam_results/eval_surprise_vs_unbounded_30s}"
FVD_CACHE_DIR="${FVD_CACHE_DIR:-$HF_HOME/memcam_fvd}"
EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-10}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p \
  "$MEMCAM_ROOT/logs" \
  "$METRICS_DIR" \
  "$FVD_CACHE_DIR" \
  "$MPLCONFIGDIR" \
  "$XDG_CACHE_HOME"
cd "$MEMCAM_ROOT"

echo "Matched Surprise Forcing versus unbounded evaluation"
echo "Manifest: $SURPRISE_MANIFEST"
echo "Surprise videos: $SURPRISE_VIDEOS"
echo "Unbounded videos: $UNBOUNDED_VIDEOS"
echo "Metrics: $METRICS_DIR"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

timeout 180s python -c \
  'import torch; assert torch.cuda.is_available(); print("CUDA:", torch.cuda.get_device_name(0))'

if ! cmp -s "$SURPRISE_MANIFEST" "$ATTENTION_MANIFEST"; then
  echo "The Surprise and attention manifests are not identical; refusing an unmatched comparison." >&2
  exit 2
fi

for video_dir in "$SURPRISE_VIDEOS" "$UNBOUNDED_VIDEOS"; do
  count=$(find "$video_dir" -maxdepth 1 -type f -name '*_30s_custom.mp4' | wc -l)
  echo "$video_dir: $count/$EXPECTED_VIDEOS videos"
  if [ "$count" -ne "$EXPECTED_VIDEOS" ]; then
    echo "Expected $EXPECTED_VIDEOS videos in $video_dir, found $count." >&2
    exit 2
  fi
done

evaluate_run () {
  run_name="$1"
  video_dir="$2"
  summary="$METRICS_DIR/$run_name/summary.json"

  if [ -f "$summary" ] && [ "$OVERWRITE" != "1" ]; then
    echo "Skipping completed metrics: $summary"
    return
  fi

  echo "=== Evaluating $run_name ==="
  python utils/evaluate_context_memory_prefix_curves.py \
    --manifest "$SURPRISE_MANIFEST" \
    --model_output_dir "$video_dir" \
    --metrics_dir "$METRICS_DIR" \
    --run_name "$run_name" \
    --source_duration 30 \
    --eval_durations 10,20,30 \
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
    --fvd_cache_dir "$FVD_CACHE_DIR" \
    --strict
}

evaluate_run surprise_b32_value "$SURPRISE_VIDEOS"
evaluate_run unbounded_matched "$UNBOUNDED_VIDEOS"

for run_name in surprise_b32_value unbounded_matched; do
  summary="$METRICS_DIR/$run_name/summary.json"
  python -c \
    'import json, sys; data=json.load(open(sys.argv[1])); print("===", sys.argv[2], "==="); [print("{}s LPIPS={:.6f} FVD={:.6f}".format(duration, row["lpips_alex"], row["fvd"])) for duration, row in data["by_duration"].items()]' \
    "$summary" "$run_name"
done

echo "Finished: $(date)"
echo "Summaries: $METRICS_DIR/{surprise_b32_value,unbounded_matched}/summary.json"
