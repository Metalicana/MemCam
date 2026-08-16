#!/bin/bash
#SBATCH --job-name=memcam_slamri_full15
#SBATCH --output=logs/memcam_slamri_full15_%j.out
#SBATCH --error=logs/memcam_slamri_full15_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc33
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=36:00:00

# One job, start to finish, for the new slam_ri_blend policy:
#   1) generate all 15 of the 60s-duration benchmark rows
#      (3,8,13,18,23,28,33,38,43,48,53,58,63,68,73 -- see docs/cluster_runbook.md)
#   2) LPIPS + FVD prefix-duration curves
#   3) VBench (custom_input)
#   4) VBench-Long (long_custom_input)
#   5) CUT3R camera-pose reconstruction + metrics
#
# Steps 3-4 need the separate `vbench` conda env (transformers version pin
# conflicts with `memcam`); everything else runs in `memcam`. This script
# switches envs twice -- `conda activate` works fine under set -euo pipefail
# on this cluster (every other job here already does one activation under
# the same flags), and re-activating over an already-active env just swaps
# it, no explicit deactivate needed.

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_matplotlib_cache_$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/memcam_xdg_cache_$USER}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/context_memory_60s}"
STEPS="${STEPS:-50}"
GPU="${GPU:-0}"
BUDGET="${BUDGET:-32}"

# beta*norm(SLAM) + (1-beta)*norm(RI). 0.5 is the untested midpoint -- this
# job is the first real test of whether the blend beats either constituent.
SLAMRI_BETA="${SLAMRI_BETA:-0.5}"
# k for the RI half's rarity-clustering threshold (same knob as MCE's
# --mce_rarity_neighbors / estimate_cluster_threshold's k-th nearest
# neighbor). 3 is the project default post-fix.
SLAMRI_RARITY_NEIGHBORS="${SLAMRI_RARITY_NEIGHBORS:-3}"

# All 15 of the 60s-duration rows -- the complete benchmark subset, not a
# further-reduced pilot (see docs/cluster_runbook.md). Override ROWS="" to
# fall back to the full multi-duration manifest filtered only by --durations 60,
# which resolves to the same 15 rows anyway.
ROWS="${ROWS:-3,8,13,18,23,28,33,38,43,48,53,58,63,68,73}"

BETA_TAG="${SLAMRI_BETA/./p}"
RUN_NAME="${RUN_NAME:-slamri_b${BUDGET}_beta${BETA_TAG}}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"

# -- Step 2 (LPIPS/FVD) settings --
# evaluate_context_memory_prefix_curves.py appends /$RUN_NAME to this itself
# (run_metrics_dir = metrics_dir / run_name) -- pass the parent, not the leaf.
MEMCAM_RESULTS_ROOT="${MEMCAM_RESULTS_ROOT:-$HOME/memcam_results}"
LPIPS_FVD_METRICS_ROOT="${LPIPS_FVD_METRICS_ROOT:-$MEMCAM_RESULTS_ROOT/eval_prefix_duration_curves_60s_b${BUDGET}}"
EVAL_DURATIONS="${EVAL_DURATIONS:-10,20,30,60}"
LEARNED_METRICS="${LEARNED_METRICS:-lpips,fvd}"
FVD_CACHE_DIR="${FVD_CACHE_DIR:-$HF_HOME/memcam_fvd}"

# -- Step 3-4 (VBench / VBench-Long) settings --
VBENCH_ROOT="${VBENCH_ROOT:-$HOME/VBench}"
VBENCH_OUTPUT_ROOT="${VBENCH_OUTPUT_ROOT:-$HOME/memcam_results/vbench_results}"
VBENCH_LONG_OUTPUT_ROOT="${VBENCH_LONG_OUTPUT_ROOT:-$HOME/memcam_results/vbench_long_results}"
VBENCH_DIMENSIONS="subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality"

# -- Step 5 (CUT3R) settings --
CUT3R_MODEL="${CUT3R_MODEL:-$MEMCAM_ROOT/CUT3R/src/cut3r_512_dpt_4_64.pth}"
CUT3R_SIZE="${CUT3R_SIZE:-512}"
CUT3R_FRAME_STRIDE="${CUT3R_FRAME_STRIDE:-30}"
CUT3R_MAX_FRAMES="${CUT3R_MAX_FRAMES:-120}"
CUT3R_OUT_DIR="${CUT3R_OUT_DIR:-$OUTPUT_ROOT/cut3r_pose_recon}"
CUT3R_METRICS_DIR="${CUT3R_METRICS_DIR:-$OUTPUT_ROOT/cut3r_camera_metrics}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR" "$OUTPUT_DIR/profiles" \
  "$LPIPS_FVD_METRICS_ROOT" "$CUT3R_OUT_DIR" "$CUT3R_METRICS_DIR"
cd "$MEMCAM_ROOT"

echo "============================================================"
echo "MemCam SLAM+RI blend -- full 15-video generation + all metrics"
echo "============================================================"
echo "Run name: $RUN_NAME"
echo "Policy: slam_ri_blend"
echo "Budget: $BUDGET"
echo "Beta: $SLAMRI_BETA"
echo "Rarity neighbors: $SLAMRI_RARITY_NEIGHBORS"
echo "Rows: $ROWS"
echo "Manifest: $MANIFEST"
echo "Output dir: $OUTPUT_DIR"
echo "Steps: $STEPS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

timeout 90s python -c \
  'import torch; assert torch.cuda.is_available(); print("CUDA:", torch.cuda.get_device_name(0))'
python utils/check_memory_policies.py

echo
echo "------------------------------------------------------------"
echo "[1/5] Generation ($(date))"
echo "------------------------------------------------------------"

python utils/run_context_memory_batch.py \
  --manifest "$MANIFEST" \
  --gpu "$GPU" \
  --durations 60 \
  --rows "$ROWS" \
  --output_dir "$OUTPUT_DIR" \
  --num_inference_steps "$STEPS" \
  --memory_policy slam_ri_blend \
  --memory_budget "$BUDGET" \
  --memory_bank_device cpu \
  --slamri_beta "$SLAMRI_BETA" \
  --slamri_rarity_neighbors "$SLAMRI_RARITY_NEIGHBORS" \
  --profile_dir "$OUTPUT_DIR/profiles"

echo "Generated videos: $(find "$OUTPUT_DIR" -maxdepth 1 -name '*_60s_custom.mp4' | wc -l)"
echo "Access traces: $(find "$OUTPUT_DIR/access_traces" -name '*.jsonl' 2>/dev/null | wc -l)"

echo
echo "------------------------------------------------------------"
echo "[2/5] LPIPS + FVD prefix-duration curves ($(date))"
echo "------------------------------------------------------------"

python utils/evaluate_context_memory_prefix_curves.py \
  --manifest "$MANIFEST" \
  --model_output_dir "$OUTPUT_DIR" \
  --metrics_dir "$LPIPS_FVD_METRICS_ROOT" \
  --run_name "$RUN_NAME" \
  --rows "$ROWS" \
  --source_duration 60 \
  --eval_durations "$EVAL_DURATIONS" \
  --learned_metrics "$LEARNED_METRICS" \
  --metric_device cuda \
  --metric_batch_size 8 \
  --learned_image_size 224 \
  --frame_stride 30 \
  --fvd_clip_length 16 \
  --fvd_clips_per_video 4 \
  --fvd_frame_stride 4 \
  --fvd_image_size 224 \
  --fvd_backend styleganv_i3d \
  --fvd_cache_dir "$FVD_CACHE_DIR"

echo "LPIPS/FVD summary: $LPIPS_FVD_METRICS_ROOT/$RUN_NAME/summary.json"

echo
echo "------------------------------------------------------------"
echo "[3/5] VBench (custom_input) ($(date))"
echo "------------------------------------------------------------"

conda activate vbench
cd "$VBENCH_ROOT"

python evaluate.py \
  --dimension $VBENCH_DIMENSIONS \
  --videos_path "$OUTPUT_DIR/" \
  --mode custom_input \
  --output_path "$VBENCH_OUTPUT_ROOT/$RUN_NAME"

echo
echo "------------------------------------------------------------"
echo "[4/5] VBench-Long (long_custom_input) ($(date))"
echo "------------------------------------------------------------"

python vbench2_beta_long/eval_long.py \
  --videos_path "$OUTPUT_DIR/" \
  --dimension $VBENCH_DIMENSIONS \
  --mode long_custom_input \
  --dev_flag \
  --output_path "$VBENCH_LONG_OUTPUT_ROOT/$RUN_NAME"

echo
echo "------------------------------------------------------------"
echo "[5/5] CUT3R reconstruction + camera metrics ($(date))"
echo "------------------------------------------------------------"

conda activate memcam
cd "$MEMCAM_ROOT"

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
echo "============================================================"
echo "All done: $(date)"
echo "============================================================"
echo "Videos:            $OUTPUT_DIR"
echo "LPIPS/FVD summary: $LPIPS_FVD_METRICS_ROOT/$RUN_NAME/summary.json"
echo "VBench:            $VBENCH_OUTPUT_ROOT/$RUN_NAME"
echo "VBench-Long:       $VBENCH_LONG_OUTPUT_ROOT/$RUN_NAME"
echo "CUT3R summary:     $CUT3R_METRICS_DIR/cut3r_camera_summary.csv"
echo
echo "LPIPS/FVD:"
cat "$LPIPS_FVD_METRICS_ROOT/$RUN_NAME/summary.json" 2>/dev/null || echo "  (not found)"
echo
echo "CUT3R:"
cat "$CUT3R_METRICS_DIR/cut3r_camera_summary.csv" 2>/dev/null || echo "  (not found)"
