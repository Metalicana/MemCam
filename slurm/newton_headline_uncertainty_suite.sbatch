#!/bin/bash
#SBATCH --job-name=keep_ci_all
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=keep_ci_all_%j.out
#SBATCH --error=keep_ci_all_%j.err

set -euo pipefail
module load cuda/cuda-12.1.0
module load ffmpeg
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
PY="${MEMCAM_ENV_PATH:-$HOME/.conda/envs/memcam}/bin/python"
export PATH="$(dirname "$PY"):$PATH"
cd "${MEMCAM_ROOT:-$HOME/MemCam}"
timeout 180s "$PY" -u -c 'import torch; torch.cuda.init(); assert torch.cuda.device_count() == 1; x = torch.ones(1, device="cuda"); x.mul_(2); torch.cuda.synchronize(); assert x.item() == 2; print(torch.cuda.get_device_name(0), flush=True)'
exec "$PY" -u paper/run_headline_uncertainty_suite.py run "$@"
