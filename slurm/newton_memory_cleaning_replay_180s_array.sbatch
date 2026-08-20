#!/bin/bash
#SBATCH --job-name=memcam_clean_mem
#SBATCH --output=logs/memcam_clean_mem_%A_%a.out
#SBATCH --error=logs/memcam_clean_mem_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc33,evc40,evc44
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%1

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate memcam

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
ROOT="${ROOT:-$HOME/memcam_results/context_180s}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory_180s/manifest.jsonl}"
DECOMP="${DECOMP:-$ROOT/unbounded_failure_decomposition_180s}"
PLAN="${PLAN:-$DECOMP/memory_cleaning_replay_plan.csv}"
REPLAY_ROOT="${REPLAY_ROOT:-$ROOT/memory_cleaning_replays_180s}"
STEPS="${STEPS:-50}"
SEED="${SEED:-42}"
CASE_INDEX="${CASE_INDEX:-$SLURM_ARRAY_TASK_ID}"

mkdir -p "$MEMCAM_ROOT/logs" "$REPLAY_ROOT"
cd "$MEMCAM_ROOT"

if [ ! -s "$PLAN" ]; then
  echo "Missing or empty replay plan: $PLAN" >&2
  exit 1
fi

CASE_COUNT=$(python -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))' "$PLAN")
if [ "$CASE_INDEX" -ge "$CASE_COUNT" ]; then
  echo "Case index $CASE_INDEX is outside the $CASE_COUNT planned cases; nothing to run."
  exit 0
fi

# The first task calibrates replay reproducibility with an explicit control.
# Later tasks save compute by using the existing baseline video as control;
# evaluation still rejects a case if its pre-intervention pixels do not match.
if [ "$CASE_INDEX" -eq 0 ]; then
  BRANCHES="${BRANCHES:-control,clean_gt}"
else
  BRANCHES="${BRANCHES:-clean_gt}"
fi

echo "MemCam generated-memory cleaning replay"
echo "Case index: $CASE_INDEX"
echo "Branches: $BRANCHES"
echo "Plan: $PLAN"
echo "Replay root: $REPLAY_ROOT"
echo "Steps: $STEPS"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

timeout 90s python -c \
  'import torch; assert torch.cuda.is_available(); print("CUDA:", torch.cuda.get_device_name(0))'

python utils/run_memory_cleaning_replay_case.py \
  --manifest "$MANIFEST" \
  --plan "$PLAN" \
  --case_index "$CASE_INDEX" \
  --root "$ROOT" \
  --output_root "$REPLAY_ROOT" \
  --branches "$BRANCHES" \
  --num_inference_steps "$STEPS" \
  --seed "$SEED"

echo "Finished: $(date)"
