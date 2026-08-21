#!/bin/bash
#SBATCH --job-name=memcam_cc_gate
#SBATCH --output=logs/memcam_cc_gate_%j.out
#SBATCH --error=logs/memcam_cc_gate_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc23,evc33,evc40,evc42,evc44
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_cc_gate_mpl_$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
ROOT="${ROOT:-$HOME/memcam_results/context_180s}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory_180s/manifest.jsonl}"
RUNS="${RUNS:-baseline,slam_b32_covisibility}"
FRAME_STRIDE="${FRAME_STRIDE:-76}"
OUT="${OUT:-$ROOT/causal_consistency_gate_pilot_180s}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUT"
cd "$MEMCAM_ROOT"

echo "MemCam causal-consistency gate validation"
echo "This job validates the gate only; it does not generate policy videos."
echo "Root: $ROOT"
echo "Manifest: $MANIFEST"
echo "Runs: $RUNS"
echo "Frame stride: $FRAME_STRIDE"
echo "Output: $OUT"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

timeout 90s python -c \
  'import torch; assert torch.cuda.is_available(); print("CUDA:", torch.cuda.get_device_name(0))'

python -m unittest \
  tests.test_causal_consistency_gate \
  tests.test_frame_quality_estimators \
  -v

python utils/calibrate_causal_consistency_gate.py \
  --manifest "$MANIFEST" \
  --root "$ROOT" \
  --runs "$RUNS" \
  --duration 180 \
  --frame_stride "$FRAME_STRIDE" \
  --dino_model facebook/dinov2-base \
  --device cuda \
  --batch_size 32 \
  --pose_bins 4 \
  --bad_quantile 0.20 \
  --max_train_clean_false_reject 0.10 \
  --test_fraction 0.3333333333 \
  --split_seed 17 \
  --min_auc 0.70 \
  --min_bad_precision 0.50 \
  --min_bad_recall 0.20 \
  --max_test_clean_false_reject 0.15 \
  --min_pose_auc_gain 0.02 \
  --min_bad_parent_auc 0.60 \
  --output_dir "$OUT" \
  --strict

echo "Finished: $(date)"
echo "Report: $OUT/report.md"
cat "$OUT/report.md"
