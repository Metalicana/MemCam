#!/bin/bash
#SBATCH --job-name=memcam_60s_mce_pilot
#SBATCH --output=logs/memcam_60s_mce_pilot_%j.out
#SBATCH --error=logs/memcam_60s_mce_pilot_%j.err
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

# Eq. 6: K(q,m) = alpha*K_geo(q,m) + (1-alpha)*K_vis(q,m). 0.65 is the
# paper's stated default engineering choice, not a theoretical constant.
MCE_ALPHA="${MCE_ALPHA:-0.65}"
# lambda: historical-vs-control query weight split. Leave unset to use the
# paper's rule (1.0 with no future queries, else 0.5).
MCE_LAMBDA="${MCE_LAMBDA:-}"
# gamma: exponential horizon-decay rate for future control queries, w_h ~
# exp(-gamma*h).
MCE_GAMMA="${MCE_GAMMA:-0.25}"
# Frame stride for sampling the known future camera path into Q_ctrl. 19
# matches one predict-latent-frame's worth of pose samples per section.
MCE_QUERY_STRIDE="${MCE_QUERY_STRIDE:-19}"
# k for Q_hist's clustering threshold (estimate_cluster_threshold's k-th
# nearest-neighbor distance). Was a dead parameter before the
# estimate_cluster_threshold fix (rarity_neighbors was accepted but never
# used); higher values coarsen clusters. 3 is the historical default, which
# is what every run before the fix effectively behaved as regardless of this
# value.
MCE_RARITY_NEIGHBORS="${MCE_RARITY_NEIGHBORS:-3}"

# Pilot default: a small, scene-spread subset of the 15 60s rows
# (3,8,13,18,23,28,33,38,43,48,53,58,63,68,73 per docs/cluster_runbook.md),
# so this finishes fast before committing to a full 15-row run. Override
# with ROWS="" (or unset) for the complete 60s benchmark.
ROWS="${ROWS-3,28,53}"

OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/mce_b${BUDGET}_pilot}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR" "$OUTPUT_DIR/profiles"
cd "$MEMCAM_ROOT"

echo "MemCam 60s MCE (Marginal Coverage Eviction) pilot"
echo "Policy: mce"
echo "Budget: $BUDGET"
echo "Alpha: $MCE_ALPHA"
echo "Lambda: ${MCE_LAMBDA:-auto (1.0 / 0.5)}"
echo "Gamma: $MCE_GAMMA"
echo "Query stride: $MCE_QUERY_STRIDE"
echo "Rarity neighbors: $MCE_RARITY_NEIGHBORS"
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
  --memory_policy mce
  --memory_budget "$BUDGET"
  --memory_bank_device cpu
  --mce_alpha "$MCE_ALPHA"
  --mce_gamma "$MCE_GAMMA"
  --mce_query_stride "$MCE_QUERY_STRIDE"
  --mce_rarity_neighbors "$MCE_RARITY_NEIGHBORS"
  --profile_dir "$OUTPUT_DIR/profiles"
)
if [ -n "$MCE_LAMBDA" ]; then
  cmd+=(--mce_lambda "$MCE_LAMBDA")
fi
if [ -n "$ROWS" ]; then
  cmd+=(--rows "$ROWS")
fi

"${cmd[@]}"

echo "Finished: $(date)"
echo "Generated videos: $(find "$OUTPUT_DIR" -maxdepth 1 -name '*_60s_custom.mp4' | wc -l)"
echo "Access traces: $(find "$OUTPUT_DIR/access_traces" -name '*.jsonl' 2>/dev/null | wc -l)"
echo "Profiles: $(find "$OUTPUT_DIR/profiles" -name '*.jsonl' 2>/dev/null | wc -l)"

echo ""
echo "Next: evaluate LPIPS/FVD against RI and SLAM at the same budget with"
echo "  RUNS=\"mce_b${BUDGET}_pilot,ri_b${BUDGET}_dino_rgb,slam_b${BUDGET}_covisibility\""
echo "  sbatch slurm/newton_eval_prefix_duration_curves_60s.sbatch"
echo "(rows differ from the full RI/SLAM runs since this is a pilot subset --"
echo "restrict the eval to matching rows, or rerun this pilot with ROWS=\"\""
echo "for a fair full-manifest comparison before drawing conclusions)."
