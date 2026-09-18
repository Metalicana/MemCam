#!/bin/bash
#SBATCH --job-name=vbench_180s
#SBATCH --output=vbench_180s_%A_%a.out
#SBATCH --error=vbench_180s_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc33,evc40,evc43,evc44
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2

set -euo pipefail
module load cuda
module load ffmpeg
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${TMPDIR:-/tmp}/vbench_180s_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
cd "${MEMCAM_ROOT:-$HOME/MemCam}"
"${MEMCAM_ENV_PATH:-$HOME/.conda/envs/memcam}/bin/python" -u paper/run_vbench_180s.py \
    --task "${SLURM_ARRAY_TASK_ID}" \
    --output "${VBENCH_180S_OUTPUT:-$HOME/memcam_results/vbench_matched15_180s_${SLURM_ARRAY_JOB_ID}}" "$@"
