#!/bin/bash
#SBATCH --job-name=keepsake_weights
#SBATCH --output=keepsake_weights_%A_%a.out
#SBATCH --error=keepsake_weights_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc33,evc40,evc43,evc44
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-74%2

set -euo pipefail
module load cuda
module load ffmpeg
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
cd "${MEMCAM_ROOT:-$HOME/MemCam}"
"${MEMCAM_ENV_PATH:-$HOME/.conda/envs/memcam}/bin/python" -u paper/run_keepsake_weight_ablation.py \
    --task "$SLURM_ARRAY_TASK_ID" \
    --output "${KEEPSAKE_WEIGHT_OUTPUT:-$HOME/memcam_results/keepsake_weights_${SLURM_ARRAY_JOB_ID}}" "$@"
