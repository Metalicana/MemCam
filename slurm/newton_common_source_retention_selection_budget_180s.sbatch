#!/bin/bash
#SBATCH --job-name=memcam_ret_sel_budget
#SBATCH --output=logs/memcam_ret_sel_budget_%j.out
#SBATCH --error=logs/memcam_ret_sel_budget_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# One CPU-only analysis job. It reuses cached DINO features and existing
# access traces; it does not decode or generate videos and is not an array.

set -euo pipefail

module load anaconda
conda activate memcam

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_ret_sel_budget_mpl_$USER}"
mkdir -p "$MPLCONFIGDIR"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
ROOT="${ROOT:-$HOME/memcam_results/context_180s}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory_180s/manifest.jsonl}"
SOURCE_DECOMP="${SOURCE_DECOMP:-$ROOT/unbounded_failure_decomposition_180s}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-$SOURCE_DECOMP/feature_cache}"
OUT="${OUT:-$ROOT/common_source_retention_selection_budget_180s}"
RUNS="${RUNS:-baseline,ri_b16_dino_rgb,ri_b32_dino_rgb,ri_b64_dino_rgb,ri_b128_dino_rgb,slam_b16_covisibility,slam_b32_covisibility,slam_b64_covisibility,slam_b128_covisibility}"
TARGET_STRIDE="${TARGET_STRIDE:-19}"
LATE_SECTION="${LATE_SECTION:-35}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUT"
cd "$MEMCAM_ROOT"

echo "MemCam common-source retention/selection budget sweep"
echo "CPU-only; cached features and real traces; no generation"
echo "Runs: $RUNS"
echo "Shared content source: baseline"
echo "Feature cache: $FEATURE_CACHE_DIR"
echo "Target stride: $TARGET_STRIDE"
echo "Late section: $LATE_SECTION"
echo "Output: $OUT"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

python -m unittest discover \
  -s tests \
  -p 'test_common_source_retention_selection_budget.py' \
  -v

python -u utils/analyze_common_source_retention_selection_budget.py \
  --manifest "$MANIFEST" \
  --root "$ROOT" \
  --runs "$RUNS" \
  --reference_run baseline \
  --content_run baseline \
  --duration 180 \
  --target_stride "$TARGET_STRIDE" \
  --late_section "$LATE_SECTION" \
  --feature_cache_dir "$FEATURE_CACHE_DIR" \
  --output_dir "$OUT" \
  --strict

echo "Finished: $(date)"
echo "Report: $OUT/report.md"
echo "Whole-rollout figure: $OUT/figures/retention_selection_budget_all.png"
echo "Late-horizon figure: $OUT/figures/retention_selection_budget_late.png"
cat "$OUT/report.md"
