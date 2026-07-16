#!/bin/bash
#SBATCH --job-name=memcam_180s_final
#SBATCH --output=logs/memcam_180s_final_%j.out
#SBATCH --error=logs/memcam_180s_final_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

module load anaconda
conda activate memcam

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
ROOT="${ROOT:-$HOME/memcam_results/context_180s}"
REPORT_ROOT="${REPORT_ROOT:-$ROOT/report_180s}"

mkdir -p "$MEMCAM_ROOT/logs" "$REPORT_ROOT/final"
cd "$MEMCAM_ROOT"

python utils/build_180s_report_table.py \
  --eval_dir "$REPORT_ROOT/eval_metrics" \
  --cut3r_summary "$REPORT_ROOT/cut3r_camera_metrics/cut3r_camera_summary.csv" \
  --revisit_dir "$REPORT_ROOT/revisit_consistency_oracle50" \
  --output_dir "$REPORT_ROOT/final" \
  --strict

cat "$REPORT_ROOT/final/report.md"
