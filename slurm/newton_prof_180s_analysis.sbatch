#!/bin/bash
#SBATCH --job-name=memcam_prof_180s
#SBATCH --output=logs/memcam_prof_180s_%j.out
#SBATCH --error=logs/memcam_prof_180s_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=48:00:00

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
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory_180s/manifest.jsonl}"
ROOT="${ROOT:-$HOME/memcam_results/context_180s}"
METRICS_DIR="${METRICS_DIR:-$ROOT/eval_metrics_180s}"
OUT_DIR="${OUT_DIR:-$ROOT/prof_180s_analysis}"
ACCESS_OUT_DIR="${ACCESS_OUT_DIR:-$ROOT/access_trace_analysis_180s}"
TRACE_OUT_DIR="${TRACE_OUT_DIR:-$ROOT/trace_usefulness_analysis_180s}"
MECHANISM_OUT_DIR="${MECHANISM_OUT_DIR:-$OUT_DIR/bounded_memory_drivers}"
MECHANISM_DETAIL_OUT_DIR="${MECHANISM_DETAIL_OUT_DIR:-$OUT_DIR/memory_mechanisms}"
REVISIT_OUT_DIR="${REVISIT_OUT_DIR:-$ROOT/revisit_consistency}"
DATASET_ROOT="${DATASET_ROOT:-}"

RUNS="${RUNS:-baseline,fifo_b16,fifo_b32,fifo_b64,fifo_b128,ri_b16_dino_rgb,ri_b32_dino_rgb,ri_b64_dino_rgb,ri_b128_dino_rgb,slam_b16_covisibility,slam_b32_covisibility,slam_b64_covisibility,slam_b128_covisibility,h2o_b16_heavy,h2o_b32_heavy,h2o_b64_heavy,h2o_b128_heavy,kcenter_b16_dino_pose,kcenter_b32_dino_pose,kcenter_b64_dino_pose,kcenter_b128_dino_pose}"
BASELINE_RUN="${BASELINE_RUN:-baseline}"
REVISIT_RUNS="${REVISIT_RUNS:-$RUNS}"
DURATIONS="${DURATIONS:-180}"
ROWS="${ROWS:-}"

RUN_EVAL="${RUN_EVAL:-1}"
RUN_TRACE="${RUN_TRACE:-1}"
RUN_REPORT="${RUN_REPORT:-1}"
RUN_MECHANISM="${RUN_MECHANISM:-1}"
RUN_REVISIT="${RUN_REVISIT:-1}"
FORCE_EVAL="${FORCE_EVAL:-0}"
STRICT_ANALYSIS="${STRICT_ANALYSIS:-0}"

FRAME_STRIDE="${FRAME_STRIDE:-90}"
MAX_FRAMES="${MAX_FRAMES:-}"
LEARNED_METRICS="${LEARNED_METRICS:-lpips,dino,fvd}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-8}"
LEARNED_IMAGE_SIZE="${LEARNED_IMAGE_SIZE:-224}"
FVD_CLIP_LENGTH="${FVD_CLIP_LENGTH:-16}"
FVD_CLIPS_PER_VIDEO="${FVD_CLIPS_PER_VIDEO:-8}"
FVD_FRAME_STRIDE="${FVD_FRAME_STRIDE:-8}"
FVD_IMAGE_SIZE="${FVD_IMAGE_SIZE:-224}"
FVD_BACKEND="${FVD_BACKEND:-styleganv_i3d}"
FVD_CACHE_DIR="${FVD_CACHE_DIR:-$HF_HOME/memcam_fvd}"
TOP_K="${TOP_K:-8}"
NO_CONTACT_SHEET="${NO_CONTACT_SHEET:-0}"

REVISIT_SAMPLE_STRIDE="${REVISIT_SAMPLE_STRIDE:-60}"
REVISIT_MIN_GAP="${REVISIT_MIN_GAP:-450}"
REVISIT_EXACT_POSITION_THRESHOLD="${REVISIT_EXACT_POSITION_THRESHOLD:-0.25}"
REVISIT_EXACT_ROTATION_DEG="${REVISIT_EXACT_ROTATION_DEG:-5.0}"
REVISIT_FORWARD_AXIS="${REVISIT_FORWARD_AXIS:-x}"

mkdir -p "$MEMCAM_ROOT/logs" "$METRICS_DIR" "$OUT_DIR" "$ACCESS_OUT_DIR" "$TRACE_OUT_DIR"
mkdir -p "$MECHANISM_OUT_DIR" "$MECHANISM_DETAIL_OUT_DIR" "$REVISIT_OUT_DIR"
cd "$MEMCAM_ROOT"

echo "Professor 180s analysis"
echo "Manifest: $MANIFEST"
echo "Generated root: $ROOT"
echo "Metrics dir: $METRICS_DIR"
echo "Output dir: $OUT_DIR"
echo "Runs: $RUNS"
echo "Revisit runs: $REVISIT_RUNS"
echo "Durations: $DURATIONS"
echo "Rows: ${ROWS:-all}"
echo "Run eval: $RUN_EVAL"
echo "Run trace: $RUN_TRACE"
echo "Run report: $RUN_REPORT"
echo "Run mechanism: $RUN_MECHANISM"
echo "Run revisit: $RUN_REVISIT"
echo "Frame stride: $FRAME_STRIDE"
echo "Max frames: ${MAX_FRAMES:-none}"
echo "Learned metrics: $LEARNED_METRICS"
echo "FVD clips/video: $FVD_CLIPS_PER_VIDEO"
echo "FVD frame stride: $FVD_FRAME_STRIDE"
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
      --durations "$DURATIONS"
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
    if [ -n "$ROWS" ]; then
      cmd+=(--rows "$ROWS")
    fi

    "${cmd[@]}"
  done
fi

STRICT_ARGS=()
if [ "$STRICT_ANALYSIS" = "1" ]; then
  STRICT_ARGS+=(--strict)
fi

ROW_ARGS=()
if [ -n "$ROWS" ]; then
  ROW_ARGS+=(--rows "$ROWS")
fi

if [ "$RUN_TRACE" = "1" ]; then
  echo
  echo "Refreshing 180s access and trace summaries"
  python utils/analyze_access_traces.py \
    --root "$ROOT" \
    --runs "$RUNS" \
    --durations "$DURATIONS" \
    --require_common_rows \
    --output_dir "$ACCESS_OUT_DIR" \
    "${ROW_ARGS[@]}" \
    "${STRICT_ARGS[@]}"

  python utils/analyze_trace_usefulness.py \
    --root "$ROOT" \
    --baseline_run "$BASELINE_RUN" \
    --runs "$RUNS" \
    --durations "$DURATIONS" \
    --require_common_targets \
    --output_dir "$TRACE_OUT_DIR" \
    "${ROW_ARGS[@]}" \
    "${STRICT_ARGS[@]}"
fi

if [ "$RUN_REPORT" = "1" ]; then
  echo
  echo "Building 180s quality/memory report"
  REPORT_ROOT="$ROOT/prof_180s_analysis_inputs"
  mkdir -p "$REPORT_ROOT"
  for link_name in access_trace_analysis trace_usefulness_analysis; do
    link_path="$REPORT_ROOT/$link_name"
    if [ -L "$link_path" ]; then
      rm -f "$link_path"
    elif [ -e "$link_path" ]; then
      echo "Refusing to overwrite non-symlink: $link_path"
      exit 1
    fi
  done
  ln -s "$ACCESS_OUT_DIR" "$REPORT_ROOT/access_trace_analysis"
  ln -s "$TRACE_OUT_DIR" "$REPORT_ROOT/trace_usefulness_analysis"

  report_cmd=(
    python utils/build_prof_60s_analysis.py
    --root "$REPORT_ROOT"
    --metrics_dir "$METRICS_DIR"
    --runs "$RUNS"
    --baseline_run "$BASELINE_RUN"
    --duration 180
    --output_dir "$OUT_DIR"
    --manifest "$MANIFEST"
    --top_k "$TOP_K"
  )

  if [ -n "$DATASET_ROOT" ]; then
    report_cmd+=(--dataset_root "$DATASET_ROOT")
  fi
  if [ "$NO_CONTACT_SHEET" = "1" ]; then
    report_cmd+=(--no_contact_sheet)
  fi

  "${report_cmd[@]}"
fi

if [ "$RUN_MECHANISM" = "1" ]; then
  echo
  echo "Building bounded-vs-unbounded memory driver figures"
  python utils/analyze_bounded_memory_drivers.py \
    --root "$ROOT" \
    --metrics_dir "$METRICS_DIR" \
    --runs "$RUNS" \
    --baseline_run "$BASELINE_RUN" \
    --duration 180 \
    --durations "$DURATIONS" \
    --output_dir "$MECHANISM_OUT_DIR" \
    "${ROW_ARGS[@]}" \
    "${STRICT_ARGS[@]}"

  echo
  echo "Building detailed memory mechanism figures"
  python utils/analyze_memory_mechanisms.py \
    --root "$ROOT" \
    --runs "$RUNS" \
    --baseline_run "$BASELINE_RUN" \
    --durations "$DURATIONS" \
    --output_dir "$MECHANISM_DETAIL_OUT_DIR" \
    --write_retained_long \
    "${ROW_ARGS[@]}" \
    "${STRICT_ARGS[@]}"
fi

if [ "$RUN_REVISIT" = "1" ]; then
  echo
  echo "Running 180s revisit consistency"
  revisit_cmd=(
    python utils/analyze_revisit_consistency.py
    --manifest "$MANIFEST"
    --model_root "$ROOT"
    --runs "$REVISIT_RUNS"
    --output_dir "$REVISIT_OUT_DIR"
    --durations "$DURATIONS"
    --sample_stride "$REVISIT_SAMPLE_STRIDE"
    --min_revisit_gap "$REVISIT_MIN_GAP"
    --exact_position_threshold "$REVISIT_EXACT_POSITION_THRESHOLD"
    --exact_rotation_deg "$REVISIT_EXACT_ROTATION_DEG"
    --forward_axis "$REVISIT_FORWARD_AXIS"
  )

  if [ -n "$DATASET_ROOT" ]; then
    revisit_cmd+=(--dataset_root "$DATASET_ROOT")
  fi
  if [ -n "$ROWS" ]; then
    revisit_cmd+=(--rows "$ROWS")
  fi

  "${revisit_cmd[@]}"
fi

echo
echo "Finished: $(date)"
echo "Open: $OUT_DIR/report.md"
echo "Open: $MECHANISM_OUT_DIR/report.md"
echo "Open: $MECHANISM_DETAIL_OUT_DIR/report.md"
echo "Open: $REVISIT_OUT_DIR/tables/revisit_summary.csv"
find "$OUT_DIR" "$MECHANISM_OUT_DIR" "$MECHANISM_DETAIL_OUT_DIR" "$REVISIT_OUT_DIR" -maxdepth 3 -type f | sort
