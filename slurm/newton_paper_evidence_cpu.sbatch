#!/bin/bash
#SBATCH --job-name=keep_evidence
#SBATCH --output=keep_evidence_%j.out
#SBATCH --error=keep_evidence_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00

set -euo pipefail
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
printf 'Job=%s Node=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST"
"${MEMCAM_ENV_PATH:-$HOME/.conda/envs/memcam}/bin/python" -u paper/run_paper_evidence.py "$@"
