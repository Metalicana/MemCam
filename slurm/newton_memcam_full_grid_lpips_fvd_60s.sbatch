#!/bin/bash
#SBATCH --job-name=memcam_lpfvd_grid
#SBATCH --output=logs/memcam_lpfvd_grid_%j.out
#SBATCH --error=logs/memcam_lpfvd_grid_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate memcam

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/memcam_results/context_memory_60s}"
METRICS_DIR="${METRICS_DIR:-$HOME/memcam_results/metrics/lpips_fvd_full_grid_60s_n15}"
ROWS="${ROWS:-3,8,13,18,23,28,33,38,43,48,53,58,63,68,73}"
RUNS="${RUNS:-baseline,fifo_b16,fifo_b32,fifo_b64,fifo_b128,ri_b16_dino_rgb,ri_b32_dino_rgb,ri_b64_dino_rgb,ri_b128_dino_rgb,slam_b16_covisibility,slam_b32_covisibility,slam_b64_covisibility,slam_b128_covisibility,kcenter_b16,kcenter_b32,kcenter_b64,kcenter_b128,mce_b16_lambda1_pilot,mce_b32_lambda1_pilot,mce_b64_lambda1_pilot,mce_b128_lambda1_pilot}"

export MEMCAM_ROOT MODEL_ROOT METRICS_DIR ROWS RUNS
export EVAL_DURATIONS="${EVAL_DURATIONS:-10,20,30,60}"
export LEARNED_METRICS="${LEARNED_METRICS:-lpips,fvd}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export FVD_CACHE_DIR="${FVD_CACHE_DIR:-$HF_HOME/memcam_fvd}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$MEMCAM_ROOT/logs" "$METRICS_DIR" "$HF_HOME"
cd "$MEMCAM_ROOT"

echo "MemCam exact-15 LPIPS/FVD full grid"
echo "Runs: $RUNS"
echo "Rows: $ROWS"
echo "Output: $METRICS_DIR"
bash scripts/evaluate_prefix_duration_curves_60s.sh
