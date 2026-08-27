#!/bin/bash
#SBATCH --job-name=memcam_fixed_pool
#SBATCH --output=logs/memcam_fixed_pool_%j.out
#SBATCH --error=logs/memcam_fixed_pool_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00

# CPU-only fixed-history candidate-admission intervention. One job, no array,
# no dependency, and no video generation.

set -euo pipefail

module load ffmpeg
module load anaconda
conda activate memcam

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_fixed_pool_mpl_$USER}"
mkdir -p "$MPLCONFIGDIR"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
ROOT="${ROOT:-$HOME/memcam_results/context_180s}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory_180s/manifest.jsonl}"
OUT="${OUT:-$ROOT/fixed_history_candidate_pools_180s}"

cd "$MEMCAM_ROOT"
mkdir -p logs "$OUT"

echo "MemCam fixed-history candidate-pool intervention"
echo "This is CPU-only; it does not generate videos."
echo "Root: $ROOT"
echo "Manifest: $MANIFEST"
echo "Output: $OUT"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

python -m unittest discover \
  -s tests \
  -p 'test_fixed_history_candidate_pools.py' \
  -v

python -u utils/analyze_fixed_history_candidate_pools.py \
  --manifest "$MANIFEST" \
  --root "$ROOT" \
  --run_name baseline \
  --duration 180 \
  --sections 35,50,65 \
  --target_offsets 19,57 \
  --pool_sizes 32,64,128,256,512,1024,all \
  --nested_repeats 4 \
  --num_samples 5000 \
  --score_batch_size 64 \
  --torch_threads 8 \
  --output_dir "$OUT" \
  --strict

echo "Finished: $(date)"
echo "Report: $OUT/report.md"
cat "$OUT/report.md"
