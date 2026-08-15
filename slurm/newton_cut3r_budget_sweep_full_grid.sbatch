#!/bin/bash
#SBATCH --job-name=cut3r_full_grid
#SBATCH --output=logs/cut3r_full_grid_%j.out
#SBATCH --error=logs/cut3r_full_grid_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00

# Reconstruction + scoring for the full budget sweep (5 policies x 4 budgets,
# 15 rows each). Idempotent -- run_cut3r_context_memory.py skips any (run,
# row) that already has completed output, so the ~90 reconstructions from
# the earlier B=32-only pass are skipped automatically; only the ~225 new
# ones (B=16/64/128) actually run. Safe to resubmit as-is if this times out.

set -euo pipefail

module load cuda
module load anaconda
conda activate memcam

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/memcam_results/context_memory_60s}"
CUT3R_DIR="${CUT3R_DIR:-$RESULTS_ROOT/cut3r_pose_recon_full_grid}"
METRICS_DIR="${METRICS_DIR:-$RESULTS_ROOT/cut3r_camera_metrics_full_grid}"
ROWS="${ROWS:-3,8,13,18,23,28,33,38,43,48,53,58,63,68,73}"

RUNS="baseline,fifo_b16,fifo_b32,fifo_b64,fifo_b128,slam_b16_covisibility,slam_b32_covisibility,slam_b64_covisibility,slam_b128_covisibility,ri_b16_dino_rgb,ri_b32_dino_rgb,ri_b64_dino_rgb,ri_b128_dino_rgb,kcenter_b16,kcenter_b32,kcenter_b64,kcenter_b128,mce_b16_lambda1_pilot,mce_b32_lambda1_pilot,mce_b64_lambda1_pilot,mce_b128_lambda1_pilot"

mkdir -p "$MEMCAM_ROOT/logs" "$CUT3R_DIR" "$METRICS_DIR"
cd "$MEMCAM_ROOT"

echo "CUT3R full budget-sweep grid"
echo "Runs: $RUNS"
echo "Rows: $ROWS"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

python utils/run_cut3r_context_memory.py \
  --manifest "$MANIFEST" \
  --root "$RESULTS_ROOT" \
  --runs "$RUNS" \
  --output_dir "$CUT3R_DIR" \
  --cut3r_root "$MEMCAM_ROOT/CUT3R" \
  --model_path "$MEMCAM_ROOT/CUT3R/src/cut3r_512_dpt_4_64.pth" \
  --size 512 --device cuda --frame_stride 30 --max_frames 120 \
  --durations 60 --rows "$ROWS"

echo "Reconstruction pass finished: $(date)"

python3 -c "
import json
from collections import Counter
status = json.load(open('$CUT3R_DIR/cut3r_run_status.json'))
print(Counter(r['status'] for r in status))
print('total:', len(status))
bad = [r for r in status if r['status'] not in ('completed', 'skipped_existing')]
for r in bad[:20]:
    print(r)
"

# Scoring is pure CPU/numpy (see: does scoring require cuda -- no), cheap to
# run right here rather than as a separate job.
python utils/evaluate_cut3r_camera_metrics.py \
  --cut3r_dir "$CUT3R_DIR" \
  --output_dir "$METRICS_DIR" \
  --runs "$RUNS" \
  --durations 60

echo "Finished: $(date)"
echo "Summary: $METRICS_DIR/cut3r_camera_summary.csv"
