#!/bin/bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-baseline}"
MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-/data/ab575577/MemCam/outputs/context_memory/$RUN_NAME}"
METRICS_DIR="${METRICS_DIR:-/data/ab575577/MemCam/eval/context_memory}"
DATASET_ROOT="${DATASET_ROOT:-}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
LEARNED_METRICS="${LEARNED_METRICS:-none}"
METRIC_DEVICE="${METRIC_DEVICE:-cuda}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-8}"
LEARNED_IMAGE_SIZE="${LEARNED_IMAGE_SIZE:-224}"
FVD_CLIP_LENGTH="${FVD_CLIP_LENGTH:-16}"
FVD_CLIPS_PER_VIDEO="${FVD_CLIPS_PER_VIDEO:-4}"
FVD_FRAME_STRIDE="${FVD_FRAME_STRIDE:-1}"
FVD_IMAGE_SIZE="${FVD_IMAGE_SIZE:-224}"
FVD_BACKEND="${FVD_BACKEND:-styleganv_i3d}"
FVD_DETECTOR_PATH="${FVD_DETECTOR_PATH:-}"
FVD_DETECTOR_URL="${FVD_DETECTOR_URL:-}"
FVD_CACHE_DIR="${FVD_CACHE_DIR:-}"
FVD_ALLOW_DOWNLOAD="${FVD_ALLOW_DOWNLOAD:-1}"
FVD_PCA_DIM="${FVD_PCA_DIM:-}"
FVD_EPS="${FVD_EPS:-1e-6}"

cd "$MEMCAM_ROOT"

cmd=(
  python utils/evaluate_context_memory.py
  --manifest "$MANIFEST"
  --model_output_dir "$MODEL_OUTPUT_DIR"
  --metrics_dir "$METRICS_DIR"
  --run_name "$RUN_NAME"
  --durations 10
  --frame_stride "$FRAME_STRIDE"
  --learned_metrics "$LEARNED_METRICS"
  --metric_device "$METRIC_DEVICE"
  --metric_batch_size "$METRIC_BATCH_SIZE"
  --learned_image_size "$LEARNED_IMAGE_SIZE"
  --fvd_clip_length "$FVD_CLIP_LENGTH"
  --fvd_clips_per_video "$FVD_CLIPS_PER_VIDEO"
  --fvd_frame_stride "$FVD_FRAME_STRIDE"
  --fvd_image_size "$FVD_IMAGE_SIZE"
  --fvd_backend "$FVD_BACKEND"
  --fvd_eps "$FVD_EPS"
)

if [ -n "$FVD_DETECTOR_PATH" ]; then
  cmd+=(--fvd_detector_path "$FVD_DETECTOR_PATH")
fi
if [ -n "$FVD_DETECTOR_URL" ]; then
  cmd+=(--fvd_detector_url "$FVD_DETECTOR_URL")
fi
if [ -n "$FVD_CACHE_DIR" ]; then
  cmd+=(--fvd_cache_dir "$FVD_CACHE_DIR")
fi
if [ "$FVD_ALLOW_DOWNLOAD" = "0" ]; then
  cmd+=(--no_fvd_download)
fi
if [ -n "$FVD_PCA_DIM" ]; then
  cmd+=(--fvd_pca_dim "$FVD_PCA_DIM")
fi

if [ -n "$DATASET_ROOT" ]; then
  cmd+=(--dataset_root "$DATASET_ROOT")
fi

"${cmd[@]}"
