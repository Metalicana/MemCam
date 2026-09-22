#!/bin/bash
#SBATCH --job-name=fvd_ri_keep30
#SBATCH --output=fvd_ri_keep30_%j.out
#SBATCH --error=fvd_ri_keep30_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc30,evc33,evc40,evc43,evc44,evc45
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
cd "${MEMCAM_ROOT:-$HOME/MemCam}"
printf 'Job=%s Node=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "${CUDA_VISIBLE_DEVICES:-unset}"
"${MEMCAM_ENV_PATH:-$HOME/.conda/envs/memcam}/bin/python" -u utils/compare_fvd_matched.py \
    --output "${FVD_COMPARE_OUTPUT:-$HOME/memcam_results/fvd_ri_keepsake_30_${SLURM_JOB_ID}}" "$@"
