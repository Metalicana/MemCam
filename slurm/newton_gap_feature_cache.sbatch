#!/bin/bash
#SBATCH --job-name=gap_features
#SBATCH --output=gap_features_%j.out
#SBATCH --error=gap_features_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc33,evc40,evc43,evc44,evc50
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail
module load cuda/cuda-12.1.0
module load ffmpeg
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
unset PYTHONHOME PYTHONPATH
cd "${MEMCAM_ROOT:-$HOME/MemCam}"
"${GAP_PYTHON:-$HOME/.conda/envs/memcam/bin/python}" -u paper/cache_gap_source_features.py \
    --output "${OUTPUT_ROOT:-$HOME/memcam_results/context_memory_60s/gap_feature_cache}" "$@"
