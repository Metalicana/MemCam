#!/bin/bash
#SBATCH --job-name=fvd_ri_keep30
#SBATCH --output=fvd_ri_keep30_%j.out
#SBATCH --error=fvd_ri_keep30_%j.err
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
for name in SLURM_JOB_ID SLURM_JOB_NODELIST SLURM_JOB_GPUS SLURM_STEP_GPUS CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER; do
    printf '%s=%s\n' "$name" "${!name:-unset}"
done
module list
if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version,compute_mode --format=csv; then
        printf 'NVIDIA inventory failed; Python will record the CUDA initialization error.\n' >&2
    fi
else
    printf 'nvidia-smi is unavailable; Python will record the CUDA initialization result.\n' >&2
fi
"${MEMCAM_ENV_PATH:-$HOME/.conda/envs/memcam}/bin/python" -u utils/compare_fvd_matched.py \
    --output "${FVD_COMPARE_OUTPUT:-$HOME/memcam_results/fvd_ri_keepsake_30_${SLURM_JOB_ID}}" "$@"
