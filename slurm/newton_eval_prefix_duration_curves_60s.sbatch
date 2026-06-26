#!/bin/bash
#SBATCH --job-name=memcam_prefix_metrics
#SBATCH --output=logs/memcam_prefix_metrics_%j.out
#SBATCH --error=logs/memcam_prefix_metrics_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

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
MODEL_ROOT="${MODEL_ROOT:-$HOME/memcam_results/context_memory_60s}"
METRICS_DIR="${METRICS_DIR:-$HOME/memcam_results/eval_prefix_duration_curves_60s_b32}"
RUNS="${RUNS:-baseline,fifo_b32,slam_b32_covisibility,ri_b32_dino_rgb}"
EVAL_DURATIONS="${EVAL_DURATIONS:-10,20,30,60}"
LEARNED_METRICS="${LEARNED_METRICS:-lpips,fvd}"
FVD_CACHE_DIR="${FVD_CACHE_DIR:-$HF_HOME/memcam_fvd}"

mkdir -p "$MEMCAM_ROOT/logs" "$METRICS_DIR"
cd "$MEMCAM_ROOT"

echo "Real prefix-duration FVD/LPIPS evaluation"
echo "Started: $(date)"
echo "Model root: $MODEL_ROOT"
echo "Metrics dir: $METRICS_DIR"
echo "Runs: $RUNS"
echo "Eval durations: $EVAL_DURATIONS"
echo "Learned metrics: $LEARNED_METRICS"

bash scripts/evaluate_prefix_duration_curves_60s.sh

echo "Finished: $(date)"
