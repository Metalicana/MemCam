#!/bin/bash
#SBATCH --job-name=memcam_slamri_cut3r
#SBATCH --output=logs/memcam_slamri_cut3r_%j.out
#SBATCH --error=logs/memcam_slamri_cut3r_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc33,evc40,evc44
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00

# CUT3R-only resume for slam_ri_blend: generation, LPIPS/FVD, and VBench all
# completed cleanly (60/60 videos evaluated, 0 failed; eval_results.json
# present); VBench-Long crashed (exit 1) partway through stage 4 and never
# reached stage 5 (CUT3R), which therefore never ran. This runs only CUT3R,
# entirely independent of VBench-Long -- no dependency on it, no risk of it
# blocking this again.

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
SLAMRI_BETA="${SLAMRI_BETA:-0.5}"
ROWS="${ROWS:-3,8,13,18,23,28,33,38,43,48,53,58,63,68,73}"

BETA_TAG="${SLAMRI_BETA/./p}"
RUN_NAME="${RUN_NAME:-slamri_b${BUDGET}_beta${BETA_TAG}}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"

CUT3R_MODEL="${CUT3R_MODEL:-$MEMCAM_ROOT/CUT3R/src/cut3r_512_dpt_4_64.pth}"
CUT3R_SIZE="${CUT3R_SIZE:-512}"
CUT3R_FRAME_STRIDE="${CUT3R_FRAME_STRIDE:-30}"
CUT3R_MAX_FRAMES="${CUT3R_MAX_FRAMES:-120}"
CUT3R_OUT_DIR="${CUT3R_OUT_DIR:-$OUTPUT_ROOT/cut3r_pose_recon}"
CUT3R_METRICS_DIR="${CUT3R_METRICS_DIR:-$OUTPUT_ROOT/cut3r_camera_metrics}"

mkdir -p "$MEMCAM_ROOT/logs" "$CUT3R_OUT_DIR" "$CUT3R_METRICS_DIR"
cd "$MEMCAM_ROOT"

echo "Resuming slam_ri_blend eval: CUT3R only"
echo "Run name: $RUN_NAME"
echo "Output dir (must already have 15 videos): $OUTPUT_DIR"
echo "Rows: $ROWS"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

if [ ! -d "$OUTPUT_DIR" ] || [ -z "$(find "$OUTPUT_DIR" -maxdepth 1 -name '*_60s_custom.mp4' -print -quit)" ]; then
  echo "ERROR: no generated videos found at $OUTPUT_DIR -- run the full job first." >&2
  exit 1
fi
echo "Generated videos found: $(find "$OUTPUT_DIR" -maxdepth 1 -name '*_60s_custom.mp4' | wc -l)"

timeout 90s python -c \
  'import torch; assert torch.cuda.is_available(); print("CUDA:", torch.cuda.get_device_name(0))'

echo
echo "------------------------------------------------------------"
echo "CUT3R reconstruction + camera metrics ($(date))"
echo "------------------------------------------------------------"

python utils/run_cut3r_context_memory.py \
  --manifest "$MANIFEST" \
  --root "$OUTPUT_ROOT" \
  --runs "$RUN_NAME" \
  --output_dir "$CUT3R_OUT_DIR" \
  --cut3r_root "$MEMCAM_ROOT/CUT3R" \
  --model_path "$CUT3R_MODEL" \
  --size "$CUT3R_SIZE" \
  --device cuda \
  --frame_stride "$CUT3R_FRAME_STRIDE" \
  --max_frames "$CUT3R_MAX_FRAMES" \
  --durations 60 \
  --rows "$ROWS"

python utils/evaluate_cut3r_camera_metrics.py \
  --cut3r_dir "$CUT3R_OUT_DIR" \
  --output_dir "$CUT3R_METRICS_DIR" \
  --runs "$RUN_NAME" \
  --durations 60 \
  --rows "$ROWS"

echo
echo "Finished: $(date)"
echo "CUT3R summary: $CUT3R_METRICS_DIR/cut3r_camera_summary.csv"
cat "$CUT3R_METRICS_DIR/cut3r_camera_summary.csv" 2>/dev/null || echo "  (not found)"
