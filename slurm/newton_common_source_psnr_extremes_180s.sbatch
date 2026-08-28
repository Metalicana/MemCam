#!/bin/bash
#SBATCH --job-name=memcam_psnr_examples
#SBATCH --output=logs/memcam_psnr_examples_%j.out
#SBATCH --error=logs/memcam_psnr_examples_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# One CPU-only analysis job. It does not generate videos and is not an array.

set -euo pipefail

module load ffmpeg
module load anaconda
conda activate memcam

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_psnr_examples_mpl_$USER}"
mkdir -p "$MPLCONFIGDIR"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
ROOT="${ROOT:-$HOME/memcam_results/context_180s}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory_180s/manifest.jsonl}"
REFERENCE_RUN="${REFERENCE_RUN:-baseline}"
POLICY_RUN="${POLICY_RUN:-slam_b32_covisibility}"
CONTENT_RUN="${CONTENT_RUN:-baseline}"
MIN_SECTION="${MIN_SECTION:-35}"
MAX_SECTION="${MAX_SECTION:-69}"
TARGET_STRIDE="${TARGET_STRIDE:-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-6}"
OUT="${OUT:-$ROOT/common_source_psnr_extremes_180s}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUT"
cd "$MEMCAM_ROOT"

echo "MemCam common-source PSNR extreme-example scan"
echo "CPU-only analysis; no generation and no job array"
echo "Reference selector: $REFERENCE_RUN"
echo "Policy selector: $POLICY_RUN"
echo "Shared image source: $CONTENT_RUN"
echo "Sections: $MIN_SECTION..$MAX_SECTION"
echo "Target stride: $TARGET_STRIDE"
echo "Output: $OUT"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

python -m unittest discover \
  -s tests \
  -p 'test_common_source_psnr_extremes.py' \
  -v

python -u utils/visualize_common_source_psnr_extremes.py \
  --manifest "$MANIFEST" \
  --root "$ROOT" \
  --reference_run "$REFERENCE_RUN" \
  --policy_run "$POLICY_RUN" \
  --content_run "$CONTENT_RUN" \
  --duration 180 \
  --min_section "$MIN_SECTION" \
  --max_section "$MAX_SECTION" \
  --target_stride "$TARGET_STRIDE" \
  --max_examples "$MAX_EXAMPLES" \
  --per_row 1 \
  --min_target_gap 152 \
  --min_psnr_delta 4.0 \
  --min_ssim_delta 0.05 \
  --max_unbounded_psnr 12.0 \
  --min_geocov_psnr 14.0 \
  --min_overlap 0.8 \
  --decode_timeout_sec 900 \
  --output_dir "$OUT" \
  --strict_thresholds \
  --strict

echo "Finished: $(date)"
echo "Overview: $OUT/figures/01_extreme_common_source_examples.png"
echo "Report: $OUT/report.md"
cat "$OUT/report.md"
