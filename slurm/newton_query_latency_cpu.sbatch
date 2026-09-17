#!/bin/bash
#SBATCH --job-name=query_latency
#SBATCH --output=query_latency_%j.out
#SBATCH --error=query_latency_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
unset PYTHONHOME PYTHONPATH
cd "${MEMCAM_ROOT:-$HOME/MemCam}"
PYTHON="${LATENCY_PYTHON:-$HOME/.conda/envs/memcam/bin/python}"
"$PYTHON" -u paper/benchmark_query_latency.py \
    --output "${OUTPUT_ROOT:-$HOME/memcam_results/query_latency_${SLURM_JOB_ID}}" "$@"
