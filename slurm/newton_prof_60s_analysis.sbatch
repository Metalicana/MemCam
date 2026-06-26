#!/bin/bash
#SBATCH --job-name=memcam_prof_60s
#SBATCH --output=logs/memcam_prof_60s_%j.out
#SBATCH --error=logs/memcam_prof_60s_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate memcam

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_matplotlib_cache_$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/memcam_xdg_cache_$USER}"
mkdir -p "$HF_HOME"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$XDG_CACHE_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
ROOT="${ROOT:-$HOME/memcam_results/context_memory_60s}"
METRICS_DIR="${METRICS_DIR:-$ROOT/eval_metrics_60s}"
OUT_DIR="${OUT_DIR:-$ROOT/prof_60s_analysis}"
MECHANISM_OUT_DIR="${MECHANISM_OUT_DIR:-$OUT_DIR/bounded_memory_drivers}"
DATASET_ROOT="${DATASET_ROOT:-}"

RUNS="${RUNS:-baseline,fifo_b32,fifo_b64,ri_b32_dino_rgb,ri_b64_dino_rgb,slam_b16_covisibility,slam_b32_covisibility,slam_b64_covisibility}"
BASELINE_RUN="${BASELINE_RUN:-baseline}"

RUN_EVAL="${RUN_EVAL:-1}"
FORCE_EVAL="${FORCE_EVAL:-0}"
FRAME_STRIDE="${FRAME_STRIDE:-30}"
MAX_FRAMES="${MAX_FRAMES:-}"
LEARNED_METRICS="${LEARNED_METRICS:-lpips,fvd}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-8}"
LEARNED_IMAGE_SIZE="${LEARNED_IMAGE_SIZE:-224}"
FVD_CLIP_LENGTH="${FVD_CLIP_LENGTH:-16}"
FVD_CLIPS_PER_VIDEO="${FVD_CLIPS_PER_VIDEO:-4}"
FVD_FRAME_STRIDE="${FVD_FRAME_STRIDE:-4}"
FVD_IMAGE_SIZE="${FVD_IMAGE_SIZE:-224}"
FVD_BACKEND="${FVD_BACKEND:-styleganv_i3d}"
FVD_CACHE_DIR="${FVD_CACHE_DIR:-$HF_HOME/memcam_fvd}"
TOP_K="${TOP_K:-8}"

mkdir -p "$MEMCAM_ROOT/logs" "$METRICS_DIR" "$OUT_DIR" "$MECHANISM_OUT_DIR"
cd "$MEMCAM_ROOT"

echo "Professor 60s analysis"
echo "Manifest: $MANIFEST"
echo "Generated root: $ROOT"
echo "Metrics dir: $METRICS_DIR"
echo "Output dir: $OUT_DIR"
echo "Mechanism output dir: $MECHANISM_OUT_DIR"
echo "Runs: $RUNS"
echo "Run eval: $RUN_EVAL"
echo "Force eval: $FORCE_EVAL"
echo "Learned metrics: $LEARNED_METRICS"
echo "Frame stride: $FRAME_STRIDE"
echo "FVD backend: $FVD_BACKEND"
echo "Started: $(date)"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

IFS=',' read -r -a RUN_ARRAY <<< "$RUNS"

if [ "$RUN_EVAL" = "1" ]; then
  for run in "${RUN_ARRAY[@]}"; do
    run="${run//[[:space:]]/}"
    [ -z "$run" ] && continue

    if [ "$FORCE_EVAL" = "0" ] && [ -f "$METRICS_DIR/$run/summary.json" ] && [ -f "$METRICS_DIR/$run/metrics.csv" ]; then
      echo "Skipping existing eval for $run"
      continue
    fi

    output_dir="$ROOT/$run"
    echo
    echo "============================================================"
    echo "Evaluating $run"
    echo "Output dir: $output_dir"
    echo "============================================================"

    cmd=(
      python utils/evaluate_context_memory.py
      --manifest "$MANIFEST"
      --model_output_dir "$output_dir"
      --metrics_dir "$METRICS_DIR"
      --run_name "$run"
      --durations 60
      --frame_stride "$FRAME_STRIDE"
      --learned_metrics "$LEARNED_METRICS"
      --metric_device cuda
      --metric_batch_size "$METRIC_BATCH_SIZE"
      --learned_image_size "$LEARNED_IMAGE_SIZE"
      --fvd_clip_length "$FVD_CLIP_LENGTH"
      --fvd_clips_per_video "$FVD_CLIPS_PER_VIDEO"
      --fvd_frame_stride "$FVD_FRAME_STRIDE"
      --fvd_image_size "$FVD_IMAGE_SIZE"
      --fvd_backend "$FVD_BACKEND"
      --fvd_cache_dir "$FVD_CACHE_DIR"
      --write_frame_metrics
    )

    if [ -n "$MAX_FRAMES" ]; then
      cmd+=(--max_frames "$MAX_FRAMES")
    fi
    if [ -n "$DATASET_ROOT" ]; then
      cmd+=(--dataset_root "$DATASET_ROOT")
    fi

    "${cmd[@]}"
  done
fi

echo
echo "Refreshing access and trace summaries"
python utils/analyze_access_traces.py \
  --root "$ROOT" \
  --runs "$RUNS" \
  --durations 60 \
  --require_common_rows \
  --output_dir "$ROOT/access_trace_analysis_prof"

python utils/analyze_trace_usefulness.py \
  --root "$ROOT" \
  --baseline_run "$BASELINE_RUN" \
  --runs "$RUNS" \
  --durations 60 \
  --require_common_targets \
  --output_dir "$ROOT/trace_usefulness_analysis_prof"

echo
echo "Building professor-ready folder"
REPORT_ROOT="$ROOT/prof_60s_analysis_inputs"
mkdir -p "$REPORT_ROOT"
for link_name in access_trace_analysis_complete15 trace_usefulness_analysis; do
  link_path="$REPORT_ROOT/$link_name"
  if [ -L "$link_path" ]; then
    rm -f "$link_path"
  elif [ -e "$link_path" ]; then
    echo "Refusing to overwrite non-symlink: $link_path"
    exit 1
  fi
done
ln -s "$ROOT/access_trace_analysis_prof" "$REPORT_ROOT/access_trace_analysis_complete15"
ln -s "$ROOT/trace_usefulness_analysis_prof" "$REPORT_ROOT/trace_usefulness_analysis"

report_cmd=(
  python utils/build_prof_60s_analysis.py
  --root "$REPORT_ROOT"
  --metrics_dir "$METRICS_DIR"
  --runs "$RUNS"
  --baseline_run "$BASELINE_RUN"
  --duration 60
  --output_dir "$OUT_DIR"
  --manifest "$MANIFEST"
  --top_k "$TOP_K"
)

if [ -n "$DATASET_ROOT" ]; then
  report_cmd+=(--dataset_root "$DATASET_ROOT")
fi

"${report_cmd[@]}"

echo
echo "Building bounded-vs-unbounded memory driver figures"
mechanism_cmd=(
  python utils/analyze_bounded_memory_drivers.py
  --root "$ROOT"
  --metrics_dir "$METRICS_DIR"
  --runs "$RUNS"
  --baseline_run "$BASELINE_RUN"
  --duration 60
  --output_dir "$MECHANISM_OUT_DIR"
)

"${mechanism_cmd[@]}"

echo
echo "Finished: $(date)"
echo "Open: $OUT_DIR/report.md"
echo "Open: $MECHANISM_OUT_DIR/report.md"
find "$OUT_DIR" -maxdepth 3 -type f | sort
