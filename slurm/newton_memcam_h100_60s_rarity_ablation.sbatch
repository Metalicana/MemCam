#!/bin/bash
#SBATCH --job-name=memcam_rarity
#SBATCH --output=logs/memcam_rarity_%j.out
#SBATCH --error=logs/memcam_rarity_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=36:00:00

# Submit this regular job once per controlled ablation:
#   POLICY_VARIANT=rarity_only
#   POLICY_VARIANT=slam_rarity_blend
# It generates 15 matched 60-second videos, then computes LPIPS/FVD, VBench,
# VBench-Long, and CUT3R. VBench-Long failure is recorded but cannot prevent
# CUT3R from running because that evaluator has failed independently before.

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_rarity_mpl_$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/memcam_rarity_xdg_$USER}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/context_memory_60s}"
METRICS_ROOT="${METRICS_ROOT:-$HOME/memcam_results/eval_prefix_duration_curves_60s_b32}"
VBENCH_ROOT="${VBENCH_ROOT:-$HOME/VBench}"
VBENCH_OUTPUT_ROOT="${VBENCH_OUTPUT_ROOT:-$HOME/memcam_results/vbench_results}"
VBENCH_LONG_OUTPUT_ROOT="${VBENCH_LONG_OUTPUT_ROOT:-$HOME/memcam_results/vbench_long_results}"
VBENCH_DIMENSIONS="subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality"
CUT3R_MODEL="${CUT3R_MODEL:-$MEMCAM_ROOT/CUT3R/src/cut3r_512_dpt_4_64.pth}"
CUT3R_OUT_DIR="${CUT3R_OUT_DIR:-$OUTPUT_ROOT/cut3r_pose_recon}"
CUT3R_METRICS_DIR="${CUT3R_METRICS_DIR:-$OUTPUT_ROOT/cut3r_camera_metrics}"

POLICY_VARIANT="${POLICY_VARIANT:?Set POLICY_VARIANT=rarity_only or slam_rarity_blend}"
ROWS="${ROWS:-3,8,13,18,23,28,33,38,43,48,53,58,63,68,73}"
STEPS="${STEPS:-50}"
BUDGET="${BUDGET:-32}"
RARITY_NEIGHBORS="${RARITY_NEIGHBORS:-3}"
SLAM_WEIGHT=0.75
GPU="${GPU:-0}"

case "$POLICY_VARIANT" in
  rarity_only)
    RUN_NAME="${RUN_NAME:-rarity_b${BUDGET}_k${RARITY_NEIGHBORS}}"
    ;;
  slam_rarity_blend)
    RUN_NAME="${RUN_NAME:-slamrarity_b${BUDGET}_s75_r25_k${RARITY_NEIGHBORS}}"
    ;;
  *)
    echo "Unsupported POLICY_VARIANT: $POLICY_VARIANT" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR" "$OUTPUT_DIR/profiles" \
  "$METRICS_ROOT" "$VBENCH_OUTPUT_ROOT/$RUN_NAME" \
  "$VBENCH_LONG_OUTPUT_ROOT/$RUN_NAME" "$CUT3R_OUT_DIR" \
  "$CUT3R_METRICS_DIR"
cd "$MEMCAM_ROOT"

echo "MemCam 60s rarity ablation"
echo "Policy: $POLICY_VARIANT"
echo "Run: $RUN_NAME"
echo "Budget: $BUDGET"
echo "Rarity neighbors: $RARITY_NEIGHBORS"
echo "Geometric weight: $SLAM_WEIGHT"
echo "Rarity weight: 0.25"
echo "Rows: $ROWS"
echo "Git commit: $(git rev-parse HEAD)"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"
echo "=== NVIDIA device ==="
nvidia-smi

python -m unittest discover -s tests -p 'test_slam_rarity_blend.py' -v

python utils/run_context_memory_batch.py \
  --manifest "$MANIFEST" \
  --gpu "$GPU" \
  --durations 60 \
  --rows "$ROWS" \
  --output_dir "$OUTPUT_DIR" \
  --num_inference_steps "$STEPS" \
  --memory_policy "$POLICY_VARIANT" \
  --memory_budget "$BUDGET" \
  --memory_bank_device cpu \
  --rarity_neighbors "$RARITY_NEIGHBORS" \
  --slamrarity_slam_weight "$SLAM_WEIGHT" \
  --profile_dir "$OUTPUT_DIR/profiles"

video_count=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*_60s_custom.mp4' | wc -l)
echo "Generated videos: $video_count/15"
if [ "$video_count" -ne 15 ]; then
  echo "Expected 15 videos before evaluation, found $video_count" >&2
  exit 1
fi

python utils/evaluate_context_memory_prefix_curves.py \
  --manifest "$MANIFEST" \
  --model_output_dir "$OUTPUT_DIR" \
  --metrics_dir "$METRICS_ROOT" \
  --run_name "$RUN_NAME" \
  --rows "$ROWS" \
  --source_duration 60 \
  --eval_durations 10,20,30,60 \
  --learned_metrics lpips,fvd \
  --metric_device cuda \
  --metric_batch_size 8 \
  --learned_image_size 224 \
  --frame_stride 30 \
  --fvd_clip_length 16 \
  --fvd_clips_per_video 4 \
  --fvd_frame_stride 4 \
  --fvd_image_size 224 \
  --fvd_backend styleganv_i3d \
  --fvd_cache_dir "$HF_HOME/memcam_fvd" \
  --strict

conda activate vbench
cd "$VBENCH_ROOT"
python evaluate.py \
  --dimension $VBENCH_DIMENSIONS \
  --videos_path "$OUTPUT_DIR/" \
  --mode custom_input \
  --output_path "$VBENCH_OUTPUT_ROOT/$RUN_NAME"

if python vbench2_beta_long/eval_long.py \
  --videos_path "$OUTPUT_DIR/" \
  --dimension $VBENCH_DIMENSIONS \
  --mode long_custom_input \
  --dev_flag \
  --output_path "$VBENCH_LONG_OUTPUT_ROOT/$RUN_NAME"; then
  echo "VBench-Long completed."
else
  echo "WARNING: VBench-Long failed; continuing to CUT3R." >&2
fi

conda activate memcam
cd "$MEMCAM_ROOT"
python utils/run_cut3r_context_memory.py \
  --manifest "$MANIFEST" \
  --root "$OUTPUT_ROOT" \
  --runs "$RUN_NAME" \
  --output_dir "$CUT3R_OUT_DIR" \
  --cut3r_root "$MEMCAM_ROOT/CUT3R" \
  --model_path "$CUT3R_MODEL" \
  --size 512 \
  --device cuda \
  --frame_stride 30 \
  --max_frames 120 \
  --durations 60 \
  --rows "$ROWS"

python utils/evaluate_cut3r_camera_metrics.py \
  --cut3r_dir "$CUT3R_OUT_DIR" \
  --output_dir "$CUT3R_METRICS_DIR" \
  --runs "$RUN_NAME" \
  --durations 60 \
  --rows "$ROWS"

echo "Finished: $(date)"
echo "Videos: $OUTPUT_DIR"
echo "LPIPS/FVD: $METRICS_ROOT/$RUN_NAME/summary.json"
echo "VBench: $VBENCH_OUTPUT_ROOT/$RUN_NAME"
echo "VBench-Long: $VBENCH_LONG_OUTPUT_ROOT/$RUN_NAME"
echo "CUT3R: $CUT3R_METRICS_DIR/cut3r_camera_summary.csv"
