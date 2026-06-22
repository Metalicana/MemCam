#!/bin/bash
#SBATCH --job-name=memcam_60s_fifo_ri
#SBATCH --output=logs/memcam_60s_fifo_ri_%j.out
#SBATCH --error=logs/memcam_60s_fifo_ri_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate memcam

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
mkdir -p "$HF_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/context_memory_60s}"
STEPS="${STEPS:-50}"
GPU="${GPU:-0}"
BUDGETS="${BUDGETS:-64,128}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_ROOT"
cd "$MEMCAM_ROOT"

echo "MemCam 60s FIFO + RI benchmark"
echo "No baseline. No SLAM."
echo "Manifest: $MANIFEST"
echo "Output root: $OUTPUT_ROOT"
echo "Budgets: $BUDGETS"
echo "Steps: $STEPS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Started: $(date)"

python - <<'PY'
import imageio.v2 as imageio
import numpy as np

frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
imageio.mimsave("/tmp/memcam_ffmpeg_slurm_check.mp4", frames, fps=2)
print("imageio ffmpeg writer ok")
PY

run_policy() {
  local policy="$1"
  local memory_policy="$2"
  local memory_budget="$3"
  local output_dir="$OUTPUT_ROOT/$policy"

  mkdir -p "$output_dir"

  echo
  echo "============================================================"
  echo "Policy: $policy"
  echo "Memory policy: $memory_policy"
  echo "Memory budget: $memory_budget"
  echo "Output dir: $output_dir"
  echo "Policy started: $(date)"
  echo "============================================================"

  python utils/run_context_memory_batch.py \
    --manifest "$MANIFEST" \
    --gpu "$GPU" \
    --durations 60 \
    --output_dir "$output_dir" \
    --num_inference_steps "$STEPS" \
    --memory_policy "$memory_policy" \
    --memory_budget "$memory_budget"

  echo "Policy finished: $(date)"
  echo "Generated 60s videos: $(find "$output_dir" -maxdepth 1 -name '*_60s_custom.mp4' | wc -l)"
  echo "Access traces: $(find "$output_dir/access_traces" -name '*.jsonl' 2>/dev/null | wc -l)"
}

IFS=',' read -r -a MEMORY_BUDGETS <<< "$BUDGETS"
for budget in "${MEMORY_BUDGETS[@]}"; do
  budget="${budget//[[:space:]]/}"
  if [ -z "$budget" ]; then
    continue
  fi

  run_policy "fifo_b${budget}" "fifo" "$budget"
  run_policy "ri_b${budget}_dino_rgb" "rarity_irreplaceability" "$budget"
done

echo
echo "FIFO + RI 60s sweep finished: $(date)"
for budget in "${MEMORY_BUDGETS[@]}"; do
  budget="${budget//[[:space:]]/}"
  if [ -z "$budget" ]; then
    continue
  fi

  for policy in "fifo_b${budget}" "ri_b${budget}_dino_rgb"; do
    output_dir="$OUTPUT_ROOT/$policy"
    echo "== $policy =="
    echo "videos: $(find "$output_dir" -maxdepth 1 -name '*_60s_custom.mp4' | wc -l)"
    echo "traces: $(find "$output_dir/access_traces" -name '*.jsonl' 2>/dev/null | wc -l)"
  done
done
