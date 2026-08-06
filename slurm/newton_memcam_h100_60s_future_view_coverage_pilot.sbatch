#!/bin/bash
#SBATCH --job-name=memcam_60s_fvc_pilot
#SBATCH --output=logs/memcam_60s_fvc_pilot_%j.out
#SBATCH --error=logs/memcam_60s_fvc_pilot_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate memcam

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
mkdir -p "$HF_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/context_memory_60s}"
BUDGET="${BUDGET:-32}"
STEPS="${STEPS:-50}"
GPU="${GPU:-0}"

# Density kernel: same DINO/RGB-weighted geometry x appearance kernel as
# density_balanced_view_coverage, reused unchanged as K(q,m) for both the
# self-covering and future-query demand pools.
FUTURE_COVERAGE_ALPHA="${FUTURE_COVERAGE_ALPHA:-0.5}"
FUTURE_COVERAGE_DINO_WEIGHT="${FUTURE_COVERAGE_DINO_WEIGHT:-0.5}"
FUTURE_COVERAGE_RGB_WEIGHT="${FUTURE_COVERAGE_RGB_WEIGHT:-0.25}"

# Prospective demand pool: sampled poses from the *known* remaining camera
# path (c2ws is the full manifest trajectory, available upfront for Task 1).
# Stride 19 matches one predict-latent-frame's worth of pose samples per
# section (76 predict frames / 4 latents); weight 1.0 gives each sampled
# future viewpoint the same demand mass as a single retained observation.
FUTURE_COVERAGE_QUERY_STRIDE="${FUTURE_COVERAGE_QUERY_STRIDE:-19}"
FUTURE_COVERAGE_QUERY_WEIGHT="${FUTURE_COVERAGE_QUERY_WEIGHT:-1.0}"

# Pilot default: a small, scene-spread subset of the 15 60s rows
# (3,8,13,18,23,28,33,38,43,48,53,58,63,68,73 per docs/cluster_runbook.md),
# so this finishes in well under an hour before committing to a full 15-row
# run. Override with ROWS="" (or unset) for the complete 60s benchmark.
ROWS="${ROWS-3,28,53}"

OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/future_view_coverage_b${BUDGET}_pilot}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR" "$OUTPUT_DIR/profiles"
cd "$MEMCAM_ROOT"

echo "MemCam 60s future-view-coverage pilot"
echo "Policy: future_view_coverage"
echo "Budget: $BUDGET"
echo "Future coverage alpha: $FUTURE_COVERAGE_ALPHA"
echo "DINO weight: $FUTURE_COVERAGE_DINO_WEIGHT"
echo "RGB weight: $FUTURE_COVERAGE_RGB_WEIGHT"
echo "Future query stride: $FUTURE_COVERAGE_QUERY_STRIDE"
echo "Future query weight: $FUTURE_COVERAGE_QUERY_WEIGHT"
echo "Manifest: $MANIFEST"
echo "Rows: ${ROWS:-all}"
echo "Output: $OUTPUT_DIR"
echo "Steps: $STEPS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

timeout 90s python -c \
  'import torch; assert torch.cuda.is_available(); print("CUDA:", torch.cuda.get_device_name(0))'
python utils/check_memory_policies.py

cmd=(
  python utils/run_context_memory_batch.py
  --manifest "$MANIFEST"
  --gpu "$GPU"
  --durations 60
  --output_dir "$OUTPUT_DIR"
  --num_inference_steps "$STEPS"
  --memory_policy future_view_coverage
  --memory_budget "$BUDGET"
  --memory_bank_device cpu
  --future_coverage_alpha "$FUTURE_COVERAGE_ALPHA"
  --future_coverage_dino_weight "$FUTURE_COVERAGE_DINO_WEIGHT"
  --future_coverage_rgb_weight "$FUTURE_COVERAGE_RGB_WEIGHT"
  --future_coverage_query_stride "$FUTURE_COVERAGE_QUERY_STRIDE"
  --future_coverage_query_weight "$FUTURE_COVERAGE_QUERY_WEIGHT"
  --profile_dir "$OUTPUT_DIR/profiles"
)
if [ -n "$ROWS" ]; then
  cmd+=(--rows "$ROWS")
fi

"${cmd[@]}"

echo "Finished: $(date)"
echo "Generated videos: $(find "$OUTPUT_DIR" -maxdepth 1 -name '*_60s_custom.mp4' | wc -l)"
echo "Access traces: $(find "$OUTPUT_DIR/access_traces" -name '*.jsonl' 2>/dev/null | wc -l)"
echo "Profiles: $(find "$OUTPUT_DIR/profiles" -name '*.jsonl' 2>/dev/null | wc -l)"
