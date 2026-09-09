#!/bin/bash
#SBATCH --job-name=memory_snowball_cpu
#SBATCH --output=logs/memory_snowball_%j.out
#SBATCH --error=logs/memory_snowball_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00

set -euo pipefail
module load ffmpeg
module load anaconda
conda activate memcam
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/memcam_results/context_memory_60s}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/snowball_60s_${SLURM_JOB_ID}}"
cd "$MEMCAM_ROOT"
python utils/visualize_memory_snowball.py \
    --manifest "$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl" \
    --root "$RESULTS_ROOT" --duration 60 --output "$OUTPUT_ROOT" --top 5
