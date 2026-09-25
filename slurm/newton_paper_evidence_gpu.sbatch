#!/bin/bash
#SBATCH --job-name=keep_fvd_ci
#SBATCH --output=keep_fvd_ci_%j.out
#SBATCH --error=keep_fvd_ci_%j.err
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00

set -euo pipefail
module load cuda/cuda-12.1.0
module load ffmpeg
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
printf 'Job=%s Node=%s CUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "${CUDA_VISIBLE_DEVICES:-unset}"
"${MEMCAM_ENV_PATH:-$HOME/.conda/envs/memcam}/bin/python" -u paper/run_paper_evidence.py "$@"
