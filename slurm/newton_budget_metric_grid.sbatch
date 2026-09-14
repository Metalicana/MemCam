#!/bin/bash
#SBATCH --job-name=budget_metrics
#SBATCH --output=logs/budget_metrics_%A_%a.out
#SBATCH --error=logs/budget_metrics_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc33,evc40,evc44
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-83%2

set -euo pipefail
module load cuda
module load ffmpeg
module load anaconda
conda activate memcam
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${TMPDIR:-/tmp}/memcam_mpl_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
cd "$MEMCAM_ROOT"
python utils/run_budget_metric_grid.py run \
    --plan "${1:?Pass the frozen plan.json}" --task "${SLURM_ARRAY_TASK_ID:?Missing array ID}"
