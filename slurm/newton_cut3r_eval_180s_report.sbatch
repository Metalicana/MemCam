#!/bin/bash
#SBATCH --job-name=memcam_cut3r_eval
#SBATCH --output=logs/memcam_cut3r_eval_%j.out
#SBATCH --error=logs/memcam_cut3r_eval_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

module load anaconda
conda activate memcam

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
ROOT="${ROOT:-$HOME/memcam_results/context_180s}"
CUT3R_DIR="${CUT3R_DIR:-$ROOT/report_180s/cut3r_pose_recon}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/report_180s/cut3r_camera_metrics}"
RUNS="${RUNS:-baseline,fifo_b16,fifo_b32,fifo_b64,fifo_b128,ri_b16_dino_rgb,ri_b32_dino_rgb,ri_b64_dino_rgb,ri_b128_dino_rgb,slam_b16_covisibility,slam_b32_covisibility,slam_b64_covisibility,slam_b128_covisibility,kcenter_b16_dino_pose,kcenter_b32_dino_pose,kcenter_b64_dino_pose,kcenter_b128_dino_pose}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR"
cd "$MEMCAM_ROOT"

python utils/evaluate_cut3r_camera_metrics.py \
  --cut3r_dir "$CUT3R_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --runs "$RUNS" \
  --durations 180

cat "$OUTPUT_DIR/cut3r_camera_summary.csv"
