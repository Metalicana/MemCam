#!/bin/bash
#SBATCH --job-name=keep_sens_cpu
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=keep_sens_cpu_%j.out
#SBATCH --error=keep_sens_cpu_%j.err

set -euo pipefail
cd "${MEMCAM_ROOT:-$HOME/MemCam}"
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
PY="${MEMCAM_ENV_PATH:-$HOME/.conda/envs/memcam}/bin/python"
test -x "$PY"
printf 'CPU-only sensitivity: job=%s node=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST"
exec "$PY" -u paper/run_keepsake_sensitivity_cpu.py "$@"
