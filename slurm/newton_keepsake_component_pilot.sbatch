#!/bin/bash
#SBATCH --job-name=keep_components
#SBATCH --output=keep_components_%j.out
#SBATCH --error=keep_components_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc27,evc30,evc33,evc38,evc40,evc43,evc44,evc45
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=15:00:00

set -euo pipefail
module load cuda/cuda-12.1.0
module load ffmpeg
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
cd "${MEMCAM_ROOT:-$HOME/MemCam}"
for name in SLURM_JOB_ID SLURM_JOB_NODELIST SLURM_JOB_GPUS CUDA_VISIBLE_DEVICES; do
    printf '%s=%s\n' "$name" "${!name:-unset}"
done
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv || true
fi
"${MEMCAM_ENV_PATH:-$HOME/.conda/envs/memcam}/bin/python" -u paper/run_keepsake_component_pilot.py \
    --output "${KEEPSAKE_PILOT_OUTPUT:-$HOME/memcam_results/keepsake_component_pilot_30s_n3}" "$@"
