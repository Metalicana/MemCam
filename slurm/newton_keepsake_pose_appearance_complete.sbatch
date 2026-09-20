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
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
cd "${MEMCAM_ROOT:-$HOME/MemCam}"
"${MEMCAM_ENV_PATH:-$HOME/.conda/envs/memcam}/bin/python" -u paper/finish_keepsake_pose_appearance.py \
    --output "${KEEPSAKE_ABLATION_OUTPUT:-$HOME/memcam_results/keepsake_pose_appearance_180s}" "$@"
