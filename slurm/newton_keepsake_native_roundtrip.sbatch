#!/bin/bash
#SBATCH --job-name=keep_native32
#SBATCH --output=keep_native32_%j.out
#SBATCH --error=keep_native32_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc27,evc30,evc33,evc40,evc43,evc44,evc45
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
module load cuda/cuda-12.1.0
module load ffmpeg
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
cd "${MEMCAM_ROOT:-$HOME/MemCam}"
printf 'Job=%s Node=%s CUDA_VISIBLE_DEVICES=%s\n' "${SLURM_JOB_ID}" "${SLURM_JOB_NODELIST}" "${CUDA_VISIBLE_DEVICES:-unset}"
exec "${MEMCAM_ENV_PATH:-$HOME/.conda/envs/memcam}/bin/python" -u utils/run_keepsake_native_roundtrip.py \
    --output "${KEEPSAKE_NATIVE_OUTPUT:-$HOME/memcam_results/keepsake_native_roundtrip_b32}" "$@"
