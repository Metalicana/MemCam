#!/bin/bash
#SBATCH --job-name=memcam_ub_decomp
#SBATCH --output=logs/memcam_ub_decomp_%j.out
#SBATCH --error=logs/memcam_ub_decomp_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc33,evc40,evc44
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate memcam

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_decomp_mpl_$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/memcam_decomp_xdg_$USER}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
ROOT="${ROOT:-$HOME/memcam_results/context_180s}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory_180s/manifest.jsonl}"
RUNS="${RUNS:-baseline,fifo_b32,ri_b32_dino_rgb,slam_b32_covisibility}"
BASELINE_RUN="${BASELINE_RUN:-baseline}"
INTERVENTION_RUN="${INTERVENTION_RUN:-slam_b32_covisibility}"
TARGET_STRIDE="${TARGET_STRIDE:-19}"
MAX_REPLAY_CASES="${MAX_REPLAY_CASES:-4}"
MIN_REPLAY_SECTION="${MIN_REPLAY_SECTION:-20}"
ROWS="${ROWS:-}"
MAX_ROWS="${MAX_ROWS:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
OUT="${OUT:-$ROOT/unbounded_failure_decomposition_180s}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUT"
cd "$MEMCAM_ROOT"

echo "MemCam 180s retention/retrieval decomposition"
echo "Root: $ROOT"
echo "Manifest: $MANIFEST"
echo "Runs: $RUNS"
echo "Intervention: $INTERVENTION_RUN"
echo "Target stride: $TARGET_STRIDE"
echo "Output: $OUT"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

timeout 90s python -c \
  'import torch; assert torch.cuda.is_available(); print("CUDA:", torch.cuda.get_device_name(0))'

cmd=(
  python utils/analyze_retrieval_quality_decomposition.py
  --manifest "$MANIFEST"
  --root "$ROOT"
  --runs "$RUNS"
  --baseline_run "$BASELINE_RUN"
  --intervention_run "$INTERVENTION_RUN"
  --duration 180
  --target_stride "$TARGET_STRIDE"
  --min_replay_section "$MIN_REPLAY_SECTION"
  --max_replay_cases "$MAX_REPLAY_CASES"
  --device cuda
  --batch_size 64
  --output_dir "$OUT"
  --strict
)

if [ -n "$ROWS" ]; then
  cmd+=(--rows "$ROWS")
fi
if [ -n "$MAX_ROWS" ]; then
  cmd+=(--max_rows "$MAX_ROWS")
fi
if [ -n "$DATASET_ROOT" ]; then
  cmd+=(--dataset_root "$DATASET_ROOT")
fi

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

echo "Finished: $(date)"
echo "Report: $OUT/report.md"
echo "Replay plan: $OUT/tables/replay_plan.csv"
cat "$OUT/report.md"
