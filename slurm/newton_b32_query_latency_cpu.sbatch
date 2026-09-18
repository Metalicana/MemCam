#!/bin/bash
#SBATCH --job-name=b32_query_latency
#SBATCH --output=b32_query_latency_%j.out
#SBATCH --error=b32_query_latency_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
unset PYTHONHOME PYTHONPATH
cd "${MEMCAM_ROOT:-$HOME/MemCam}"
"${LATENCY_PYTHON:-$HOME/.conda/envs/memcam/bin/python}" -u paper/benchmark_b32_query_latency.py \
    --output "${OUTPUT_ROOT:-$HOME/memcam_results/b32_query_latency_60s_${SLURM_JOB_ID}}" "$@"
