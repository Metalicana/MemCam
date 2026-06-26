#!/bin/bash
set -euo pipefail

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/memcam_results/context_memory_60s}"
METRICS_DIR="${METRICS_DIR:-$HOME/memcam_results/eval_prefix_duration_curves_60s}"
RUNS="${RUNS:-baseline,fifo_b32,slam_b32_covisibility,ri_b32_dino_rgb}"
SOURCE_DURATION="${SOURCE_DURATION:-60}"
EVAL_DURATIONS="${EVAL_DURATIONS:-10,20,40,60}"
LEARNED_METRICS="${LEARNED_METRICS:-fvd}"
METRIC_DEVICE="${METRIC_DEVICE:-cuda}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-8}"
LEARNED_IMAGE_SIZE="${LEARNED_IMAGE_SIZE:-224}"
FRAME_STRIDE="${FRAME_STRIDE:-30}"
FVD_CLIP_LENGTH="${FVD_CLIP_LENGTH:-16}"
FVD_CLIPS_PER_VIDEO="${FVD_CLIPS_PER_VIDEO:-4}"
FVD_FRAME_STRIDE="${FVD_FRAME_STRIDE:-4}"
FVD_IMAGE_SIZE="${FVD_IMAGE_SIZE:-224}"
FVD_BACKEND="${FVD_BACKEND:-styleganv_i3d}"
FVD_CACHE_DIR="${FVD_CACHE_DIR:-$HOME/hf_cache/memcam_fvd}"
FVD_DETECTOR_PATH="${FVD_DETECTOR_PATH:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
LIMIT="${LIMIT:-}"
ROWS="${ROWS:-}"
START_ROW="${START_ROW:-}"
END_ROW="${END_ROW:-}"
WRITE_FRAME_METRICS="${WRITE_FRAME_METRICS:-0}"

cd "$MEMCAM_ROOT"
mkdir -p "$METRICS_DIR"

echo "Prefix duration-curve evaluation"
echo "Manifest: $MANIFEST"
echo "Model root: $MODEL_ROOT"
echo "Metrics dir: $METRICS_DIR"
echo "Runs: $RUNS"
echo "Source duration: $SOURCE_DURATION"
echo "Eval durations: $EVAL_DURATIONS"
echo "Learned metrics: $LEARNED_METRICS"

IFS=',' read -r -a RUN_ARRAY <<< "$RUNS"
for run in "${RUN_ARRAY[@]}"; do
  run="${run//[[:space:]]/}"
  [ -z "$run" ] && continue

  output_dir="$MODEL_ROOT/$run"
  echo
  echo "============================================================"
  echo "Evaluating prefix curve for $run"
  echo "Output dir: $output_dir"
  echo "============================================================"

  cmd=(
    python utils/evaluate_context_memory_prefix_curves.py
    --manifest "$MANIFEST"
    --model_output_dir "$output_dir"
    --metrics_dir "$METRICS_DIR"
    --run_name "$run"
    --source_duration "$SOURCE_DURATION"
    --eval_durations "$EVAL_DURATIONS"
    --learned_metrics "$LEARNED_METRICS"
    --metric_device "$METRIC_DEVICE"
    --metric_batch_size "$METRIC_BATCH_SIZE"
    --learned_image_size "$LEARNED_IMAGE_SIZE"
    --frame_stride "$FRAME_STRIDE"
    --fvd_clip_length "$FVD_CLIP_LENGTH"
    --fvd_clips_per_video "$FVD_CLIPS_PER_VIDEO"
    --fvd_frame_stride "$FVD_FRAME_STRIDE"
    --fvd_image_size "$FVD_IMAGE_SIZE"
    --fvd_backend "$FVD_BACKEND"
    --fvd_cache_dir "$FVD_CACHE_DIR"
  )

  if [ -n "$FVD_DETECTOR_PATH" ]; then
    cmd+=(--fvd_detector_path "$FVD_DETECTOR_PATH")
  fi
  if [ -n "$DATASET_ROOT" ]; then
    cmd+=(--dataset_root "$DATASET_ROOT")
  fi
  if [ -n "$LIMIT" ]; then
    cmd+=(--limit "$LIMIT")
  fi
  if [ -n "$ROWS" ]; then
    cmd+=(--rows "$ROWS")
  fi
  if [ -n "$START_ROW" ]; then
    cmd+=(--start_row "$START_ROW")
  fi
  if [ -n "$END_ROW" ]; then
    cmd+=(--end_row "$END_ROW")
  fi
  if [ "$WRITE_FRAME_METRICS" = "1" ]; then
    cmd+=(--write_frame_metrics)
  fi

  "${cmd[@]}"
done

echo
echo "Done. Summaries:"
find "$METRICS_DIR" -maxdepth 2 -name summary.json | sort
