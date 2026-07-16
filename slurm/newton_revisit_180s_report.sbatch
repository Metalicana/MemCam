#!/bin/bash
#SBATCH --job-name=memcam_revisit_report
#SBATCH --output=logs/memcam_revisit_report_%j.out
#SBATCH --error=logs/memcam_revisit_report_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail

module load ffmpeg
module load anaconda
conda activate memcam

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_mpl_$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/memcam_xdg_$USER}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory_180s/manifest.jsonl}"
ROOT="${ROOT:-$HOME/memcam_results/context_180s}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/report_180s/revisit_consistency_oracle50}"
RUNS="${RUNS:-baseline,fifo_b16,fifo_b32,fifo_b64,fifo_b128,ri_b16_dino_rgb,ri_b32_dino_rgb,ri_b64_dino_rgb,ri_b128_dino_rgb,slam_b16_covisibility,slam_b32_covisibility,slam_b64_covisibility,slam_b128_covisibility,kcenter_b16_dino_pose,kcenter_b32_dino_pose,kcenter_b64_dino_pose,kcenter_b128_dino_pose}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR"
cd "$MEMCAM_ROOT"

python utils/analyze_revisit_consistency.py \
  --manifest "$MANIFEST" \
  --model_root "$ROOT" \
  --runs "$RUNS" \
  --output_dir "$OUTPUT_DIR" \
  --durations 180 \
  --sample_stride 30 \
  --min_revisit_gap 450 \
  --exact_position_threshold 0.25 \
  --exact_rotation_deg 5.0 \
  --revisit_types exact_pose \
  --max_gt_worst_patch_rmse 50 \
  --patch_size 96

echo "Self-consistency summary:"
cat "$OUTPUT_DIR/tables/revisit_summary.csv"
echo "Delta-vs-GT summary:"
cat "$OUTPUT_DIR/tables/revisit_delta_summary_vs_gt.csv"
