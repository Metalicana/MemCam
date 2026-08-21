#!/bin/bash
#SBATCH --job-name=memcam_iqa_cal
#SBATCH --output=logs/memcam_iqa_cal_%j.out
#SBATCH --error=logs/memcam_iqa_cal_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc23,evc33,evc40,evc42,evc44
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate memcam

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_iqa_cal_mpl_$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
ROOT="${ROOT:-$HOME/memcam_results/context_180s}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory_180s/manifest.jsonl}"
RUNS="${RUNS:-baseline,fifo_b32,ri_b32_dino_rgb,slam_b32_covisibility}"
IQA_METRICS="${IQA_METRICS:-musiq-paq2piq,clipiqa+,topiq_nr,niqe}"
FRAME_STRIDE="${FRAME_STRIDE:-30}"
QUALITY_SIZE="${QUALITY_SIZE:-224}"
OUT="${OUT:-$ROOT/frame_quality_estimator_calibration_180s}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUT"
cd "$MEMCAM_ROOT"

echo "MemCam no-reference frame-quality estimator calibration"
echo "Root: $ROOT"
echo "Manifest: $MANIFEST"
echo "Runs: $RUNS"
echo "IQA metrics: $IQA_METRICS"
echo "Frame stride: $FRAME_STRIDE"
echo "Output: $OUT"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

timeout 90s python -c \
  'import torch; assert torch.cuda.is_available(); print("CUDA:", torch.cuda.get_device_name(0))'
python -c 'import pyiqa; print("pyiqa available; models:", len(pyiqa.list_models()))'

python -m unittest tests.test_frame_quality_estimators -v

python utils/calibrate_frame_quality_estimators.py \
  --manifest "$MANIFEST" \
  --root "$ROOT" \
  --runs "$RUNS" \
  --duration 180 \
  --frame_stride "$FRAME_STRIDE" \
  --quality_size "$QUALITY_SIZE" \
  --iqa_metrics "$IQA_METRICS" \
  --device cuda \
  --batch_size 16 \
  --bad_quantile 0.20 \
  --max_train_clean_false_reject 0.10 \
  --test_fraction 0.3333333333 \
  --split_seed 17 \
  --output_dir "$OUT" \
  --strict

echo "Finished: $(date)"
echo "Report: $OUT/report.md"
cat "$OUT/report.md"
