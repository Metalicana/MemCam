#!/bin/bash
#SBATCH --job-name=budget_report
#SBATCH --output=logs/budget_report_%j.out
#SBATCH --error=logs/budget_report_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
module load anaconda
conda activate memcam
cd "${MEMCAM_ROOT:-$HOME/MemCam}"
python utils/run_budget_metric_grid.py report --plan "${1:?Pass the frozen plan.json}"
