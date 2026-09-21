#!/bin/bash
#SBATCH --job-name=keepsake_ablation
#SBATCH --output=keepsake_ablation_%j.out
#SBATCH --error=keepsake_ablation_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --exclude=evc33,evc40,evc43,evc44
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=72:00:00

set -euo pipefail
module load cuda
module load ffmpeg
unset PYTHONHOME PYTHONPATH
# Match CUDA ordinal ordering to Slurm/NVML; preserve the allocated device mask.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
for name in SLURM_JOB_ID SLURM_JOB_NODELIST SLURM_JOB_GPUS SLURM_STEP_GPUS CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER; do
    printf '%s=%s\n' "$name" "${!name:-unset}"
done
if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi --query-gpu=index,uuid,pci.bus_id,name --format=csv; then
        printf 'GPU inventory failed; the Python kernel checks will report CUDA availability.\n' >&2
    fi
fi
cd "${MEMCAM_ROOT:-$HOME/MemCam}"
"${MEMCAM_ENV_PATH:-$HOME/.conda/envs/memcam}/bin/python" -u paper/finish_keepsake_pose_appearance.py \
    --output "${KEEPSAKE_ABLATION_OUTPUT:-$HOME/memcam_results/keepsake_pose_appearance_180s}" "$@"
