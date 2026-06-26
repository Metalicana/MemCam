#!/bin/bash
#SBATCH --job-name=memcam_today_metrics
#SBATCH --output=logs/memcam_today_metrics_%j.out
#SBATCH --error=logs/memcam_today_metrics_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate memcam

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_matplotlib_cache_$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/memcam_xdg_cache_$USER}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
OUT_ROOT="${OUT_ROOT:-$HOME/memcam_results/today_fvd_lpips_prefix_60s}"
EVAL_DURATIONS="${EVAL_DURATIONS:-10,20,30,60}"
LEARNED_METRICS="${LEARNED_METRICS:-lpips,fvd}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUT_ROOT"
cd "$MEMCAM_ROOT"

echo "One-command FVD/LPIPS prefix plot job"
echo "Started: $(date)"
echo "Output root: $OUT_ROOT"
echo "Durations: $EVAL_DURATIONS"
echo "Metrics: $LEARNED_METRICS"

bash scripts/make_today_fvd_lpips_prefix_plots_60s.sh

echo "Finished: $(date)"
