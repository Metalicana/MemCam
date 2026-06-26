#!/bin/bash
set -euo pipefail

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/memcam_results/context_memory_60s}"
OUT_ROOT="${OUT_ROOT:-$HOME/memcam_results/today_fvd_lpips_prefix_60s}"
EVAL_DURATIONS="${EVAL_DURATIONS:-10,20,30,60}"
LEARNED_METRICS="${LEARNED_METRICS:-lpips,fvd}"
FVD_CLIPS_PER_VIDEO="${FVD_CLIPS_PER_VIDEO:-4}"
FVD_FRAME_STRIDE="${FVD_FRAME_STRIDE:-4}"
FRAME_STRIDE="${FRAME_STRIDE:-30}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-8}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PLOT="${RUN_PLOT:-1}"

RUNS_B32="${RUNS_B32:-baseline,fifo_b32,slam_b32_covisibility,ri_b32_dino_rgb}"
RUNS_B64="${RUNS_B64:-baseline,fifo_b64,slam_b64_covisibility,ri_b64_dino_rgb}"

METRICS_B32="$OUT_ROOT/eval_b32"
METRICS_B64="$OUT_ROOT/eval_b64"
PLOTS_B32="$OUT_ROOT/plots_b32"
PLOTS_B64="$OUT_ROOT/plots_b64"

cd "$MEMCAM_ROOT"
mkdir -p "$OUT_ROOT" "$METRICS_B32" "$METRICS_B64" "$PLOTS_B32" "$PLOTS_B64"

echo "Today FVD/LPIPS prefix plots"
echo "Model root: $MODEL_ROOT"
echo "Output root: $OUT_ROOT"
echo "Durations: $EVAL_DURATIONS"
echo "Metrics: $LEARNED_METRICS"
echo "FVD clips/video: $FVD_CLIPS_PER_VIDEO"
echo "Frame stride for LPIPS: $FRAME_STRIDE"

if [ "$RUN_EVAL" = "1" ]; then
  echo
  echo "============================================================"
  echo "Evaluating Budget 32"
  echo "============================================================"
  RUNS="$RUNS_B32" \
  MODEL_ROOT="$MODEL_ROOT" \
  METRICS_DIR="$METRICS_B32" \
  EVAL_DURATIONS="$EVAL_DURATIONS" \
  LEARNED_METRICS="$LEARNED_METRICS" \
  FVD_CLIPS_PER_VIDEO="$FVD_CLIPS_PER_VIDEO" \
  FVD_FRAME_STRIDE="$FVD_FRAME_STRIDE" \
  FRAME_STRIDE="$FRAME_STRIDE" \
  METRIC_BATCH_SIZE="$METRIC_BATCH_SIZE" \
  bash scripts/evaluate_prefix_duration_curves_60s.sh

  echo
  echo "============================================================"
  echo "Evaluating Budget 64"
  echo "============================================================"
  RUNS="$RUNS_B64" \
  MODEL_ROOT="$MODEL_ROOT" \
  METRICS_DIR="$METRICS_B64" \
  EVAL_DURATIONS="$EVAL_DURATIONS" \
  LEARNED_METRICS="$LEARNED_METRICS" \
  FVD_CLIPS_PER_VIDEO="$FVD_CLIPS_PER_VIDEO" \
  FVD_FRAME_STRIDE="$FVD_FRAME_STRIDE" \
  FRAME_STRIDE="$FRAME_STRIDE" \
  METRIC_BATCH_SIZE="$METRIC_BATCH_SIZE" \
  bash scripts/evaluate_prefix_duration_curves_60s.sh
fi

if [ "$RUN_PLOT" = "1" ]; then
  echo
  echo "============================================================"
  echo "Plotting Budget 32"
  echo "============================================================"
  python utils/plot_metric_duration_curves.py \
    --metrics_dirs "$METRICS_B32" \
    --runs "baseline=Unbounded,fifo_b32=FIFO,slam_b32_covisibility=SLAM,ri_b32_dino_rgb=Mine" \
    --durations "$EVAL_DURATIONS" \
    --metrics fvd,lpips_alex \
    --reference_run baseline \
    --output_dir "$PLOTS_B32" \
    --title_prefix "Budget 32"

  echo
  echo "============================================================"
  echo "Plotting Budget 64"
  echo "============================================================"
  python utils/plot_metric_duration_curves.py \
    --metrics_dirs "$METRICS_B64" \
    --runs "baseline=Unbounded,fifo_b64=FIFO,slam_b64_covisibility=SLAM,ri_b64_dino_rgb=Mine" \
    --durations "$EVAL_DURATIONS" \
    --metrics fvd,lpips_alex \
    --reference_run baseline \
    --output_dir "$PLOTS_B64" \
    --title_prefix "Budget 64"
fi

echo
echo "PNG outputs:"
find "$OUT_ROOT" -name '*.png' | sort
echo
echo "CSV outputs:"
find "$OUT_ROOT" -name '*.csv' | sort
