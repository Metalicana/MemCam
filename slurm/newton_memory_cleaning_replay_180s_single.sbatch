#!/bin/bash
#SBATCH --job-name=memcam_clean_one
#SBATCH --output=logs/memcam_clean_one_%j.out
#SBATCH --error=logs/memcam_clean_one_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc23,evc33,evc40,evc42,evc44
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=18:00:00

# One allocation, one sequential process chain, no array, no dependency.

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate memcam

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_clean_one_mpl_$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/memcam_clean_one_xdg_$USER}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
ROOT="${ROOT:-$HOME/memcam_results/context_180s}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory_180s/manifest.jsonl}"
DECOMP="${DECOMP:-$ROOT/unbounded_failure_decomposition_180s}"
PLAN="${PLAN:-$DECOMP/memory_cleaning_replay_plan.csv}"
REPLAY_ROOT="${REPLAY_ROOT:-$ROOT/memory_cleaning_replays_180s}"
OUT="${OUT:-$REPLAY_ROOT/evaluation}"
STEPS="${STEPS:-50}"
SEED="${SEED:-42}"

mkdir -p "$MEMCAM_ROOT/logs" "$REPLAY_ROOT" "$OUT"
cd "$MEMCAM_ROOT"

if [ ! -s "$PLAN" ]; then
  echo "Missing or empty replay plan: $PLAN" >&2
  exit 1
fi

CASE_COUNT=$(python -c \
  'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))' \
  "$PLAN")

echo "MemCam memory-cleaning replay: single sequential job"
echo "Cases: $CASE_COUNT"
echo "Plan: $PLAN"
echo "Replay root: $REPLAY_ROOT"
echo "Evaluation: $OUT"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

echo "=== NVIDIA device ==="
nvidia-smi
python -c \
  'import torch; assert torch.cuda.is_available(); print("PyTorch CUDA:", torch.cuda.get_device_name(0))'

for ((case_index=0; case_index<CASE_COUNT; case_index++)); do
  if [ "$case_index" -eq 0 ]; then
    branches="control,clean_gt"
  else
    branches="clean_gt"
  fi

  echo "=== Case $case_index/$((CASE_COUNT - 1)): $branches ==="
  python utils/run_memory_cleaning_replay_case.py \
    --manifest "$MANIFEST" \
    --plan "$PLAN" \
    --case_index "$case_index" \
    --root "$ROOT" \
    --output_root "$REPLAY_ROOT" \
    --branches "$branches" \
    --num_inference_steps "$STEPS" \
    --seed "$SEED"
done

echo "=== Evaluating all completed replay cases ==="
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
