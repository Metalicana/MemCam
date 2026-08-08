#!/bin/bash
#SBATCH --job-name=vbench_context_memory_60s
#SBATCH --output=logs/vbench_context_memory_60s_%j.out
#SBATCH --error=logs/vbench_context_memory_60s_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

module load cuda
module load anaconda
conda activate vbench

VBENCH_ROOT="${VBENCH_ROOT:-$HOME/VBench}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/memcam_results/context_memory_60s}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/vbench_results}"
DIMENSIONS="subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality"

# Idempotent by default: a run is skipped if it already has a *_eval_results.json
# from a previous attempt, so resubmitting this job after a partial/failed run
# only redoes what's missing. Override with FORCE=1 to always recompute.
FORCE="${FORCE:-0}"

RUNS="${RUNS:-baseline fifo_b32 ri_b32_dino_rgb slam_b32_covisibility future_view_coverage_b32_pilot mce_b32_pilot mce_b32_lambda08_pilot mce_b32_lambda1_pilot}"

mkdir -p "$HOME/MemCam/logs" "$OUTPUT_ROOT"
cd "$VBENCH_ROOT"

echo "VBench batch eval over context_memory_60s runs"
echo "Runs: $RUNS"
echo "Dimensions: $DIMENSIONS"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

for run in $RUNS; do
  out_dir="$OUTPUT_ROOT/$run"
  if [ "$FORCE" != "1" ] && compgen -G "$out_dir/*_eval_results.json" > /dev/null; then
    echo "[skip] $run already has eval results in $out_dir"
    continue
  fi

  video_dir="$RESULTS_ROOT/$run/"
  if [ ! -d "$video_dir" ]; then
    echo "[skip] $run -- no directory at $video_dir"
    continue
  fi

  echo "[run] $run  ($(find "$video_dir" -maxdepth 1 -name '*_60s_custom.mp4' | wc -l) videos)  $(date)"
  python evaluate.py \
    --dimension $DIMENSIONS \
    --videos_path "$video_dir" \
    --mode custom_input \
    --output_path "$out_dir"
  echo "[done] $run  $(date)"
done

echo "Finished: $(date)"
echo "Results under: $OUTPUT_ROOT"
