#!/bin/bash
#SBATCH --job-name=memcam_clean_eval
#SBATCH --output=logs/memcam_clean_eval_%j.out
#SBATCH --error=logs/memcam_clean_eval_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc33,evc40,evc44
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate memcam

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_clean_eval_mpl_$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/memcam_clean_eval_xdg_$USER}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
ROOT="${ROOT:-$HOME/memcam_results/context_180s}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory_180s/manifest.jsonl}"
DECOMP="${DECOMP:-$ROOT/unbounded_failure_decomposition_180s}"
PLAN="${PLAN:-$DECOMP/memory_cleaning_replay_plan.csv}"
REPLAY_ROOT="${REPLAY_ROOT:-$ROOT/memory_cleaning_replays_180s}"
OUT="${OUT:-$REPLAY_ROOT/evaluation}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUT"
cd "$MEMCAM_ROOT"

echo "MemCam generated-memory cleaning replay evaluation"
echo "Plan: $PLAN"
echo "Replays: $REPLAY_ROOT"
echo "Output: $OUT"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

timeout 90s python -c \
  'import torch; assert torch.cuda.is_available(); print("CUDA:", torch.cuda.get_device_name(0))'

python utils/evaluate_memory_cleaning_replays.py \
  --manifest "$MANIFEST" \
  --plan "$PLAN" \
  --root "$ROOT" \
  --baseline_run baseline \
  --replay_root "$REPLAY_ROOT" \
  --output_dir "$OUT" \
  --device cuda \
  --batch_size 16 \
  --frame_stride 4 \
  --prefix_mae_tolerance 1.0 \
  --strict

echo "Finished: $(date)"
echo "Report: $OUT/report.md"
cat "$OUT/report.md"
