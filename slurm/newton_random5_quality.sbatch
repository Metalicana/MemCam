#!/bin/bash
#SBATCH --job-name=quality_random5
#SBATCH --output=quality_random5_%j.out
#SBATCH --error=quality_random5_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc27,evc30,evc33,evc40,evc43,evc44,evc45
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail
module load cuda/cuda-12.1.0
module load ffmpeg
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "${MEMCAM_ROOT:-$HOME/MemCam}"
printf 'Job=%s Node=%s CUDA_VISIBLE_DEVICES=%s\n' "${SLURM_JOB_ID}" "${SLURM_JOB_NODELIST}" "${CUDA_VISIBLE_DEVICES:-unset}"
"${MEMCAM_ENV_PATH:-$HOME/.conda/envs/memcam}/bin/python" -u utils/evaluate_random5_quality.py \
    --output "${RANDOM5_OUTPUT:-$HOME/memcam_results/random5_quality_60s_seed0}" "$@"
