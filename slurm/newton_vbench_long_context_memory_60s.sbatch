#!/bin/bash
#SBATCH --job-name=vbench_long_60s
#SBATCH --output=logs/vbench_long_60s_%j.out
#SBATCH --error=logs/vbench_long_60s_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

# VBench-Long over context_memory_60s runs. Untested in this project before
# now -- traced vbench2_beta_long/__init__.py directly: evaluate() calls
# preprocess() automatically, which auto-splits each video into 2s clips
# under <videos_path>/split_clip/ (long_custom_input mode hardcodes 2s clip
# length), so no manual scene-splitting step is needed despite what you'd
# guess from the folder structure alone. This DOES write a split_clip/
# subfolder of re-encoded clips into each run's video directory as a side
# effect -- extra disk, not just eval output.
#
# Defaults to a ONE-run smoke test (mce_b32_lambda1_pilot) so you can sanity
# check the auto-split + dimension scoring actually works before committing
# the full grid. Override RUNS to the full list once that's confirmed:
#   RUNS="baseline fifo_b16 fifo_b32 fifo_b64 fifo_b128 slam_b16_covisibility slam_b32_covisibility slam_b64_covisibility slam_b128_covisibility ri_b16_dino_rgb ri_b32_dino_rgb ri_b64_dino_rgb ri_b128_dino_rgb kcenter_b16 kcenter_b32 kcenter_b64 kcenter_b128 mce_b16_lambda1_pilot mce_b32_lambda1_pilot mce_b64_lambda1_pilot mce_b128_lambda1_pilot" \
#   sbatch slurm/newton_vbench_long_context_memory_60s.sbatch

set -euo pipefail

module load cuda
module load anaconda
conda activate vbench

VBENCH_ROOT="${VBENCH_ROOT:-$HOME/VBench}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/memcam_results/context_memory_60s}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/vbench_long_results}"
DIMENSIONS="subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality"
FORCE="${FORCE:-0}"

RUNS="${RUNS:-mce_b32_lambda1_pilot}"

mkdir -p "$HOME/MemCam/logs" "$OUTPUT_ROOT"
cd "$VBENCH_ROOT"

echo "VBench-Long batch eval over context_memory_60s runs"
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
  python vbench2_beta_long/eval_long.py \
    --videos_path "$video_dir" \
    --dimension $DIMENSIONS \
    --mode long_custom_input \
    --dev_flag \
    --output_path "$out_dir"
  echo "[done] $run  $(date)"
done

echo "Finished: $(date)"
echo "Results under: $OUTPUT_ROOT"
