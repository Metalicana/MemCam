#!/bin/bash
#SBATCH --job-name=keep_comp_stats
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
unset PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec "${MEMCAM_ENV_PATH:-$HOME/.conda/envs/memcam}/bin/python" -u \
  paper/run_keepsake_component_study.py report "$@"
