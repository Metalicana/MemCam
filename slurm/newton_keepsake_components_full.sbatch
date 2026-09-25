#!/bin/bash
#SBATCH --job-name=keep_comp60
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail
module load cuda/cuda-12.1.0
module load ffmpeg
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
PY="${MEMCAM_ENV_PATH:-$HOME/.conda/envs/memcam}/bin/python"
export PATH="$(dirname "$PY"):$PATH"
for name in SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_JOB_NODELIST SLURM_JOB_GPUS CUDA_VISIBLE_DEVICES; do
    printf '%s=%s\n' "$name" "${!name:-unset}"
done
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    exec "$PY" -u paper/run_keepsake_component_study.py run "$@" --scene "$SLURM_ARRAY_TASK_ID"
fi
exec "$PY" -u paper/run_keepsake_component_study.py run "$@"
