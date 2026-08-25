#!/bin/bash
#SBATCH --job-name=memcam_60s_hyst
#SBATCH --output=logs/memcam_60s_hyst_%j.out
#SBATCH --error=logs/memcam_60s_hyst_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=36:00:00

# One sequential, resume-safe job: generate all 15 videos with the final
# coverage-hysteresis policy, evaluate 10/20/30/60s LPIPS and FVD, then VBench.

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_hyst_mpl_$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/memcam_hyst_xdg_$USER}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/context_memory_60s}"
METRICS_ROOT="${METRICS_ROOT:-$HOME/memcam_results/eval_prefix_duration_curves_60s_b32}"
VBENCH_ROOT="${VBENCH_ROOT:-$HOME/VBench}"
VBENCH_OUTPUT_ROOT="${VBENCH_OUTPUT_ROOT:-$HOME/memcam_results/vbench_results}"
VBENCH_DIMENSIONS="subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality"
ROWS="${ROWS:-3,8,13,18,23,28,33,38,43,48,53,58,63,68,73}"
RUN_NAME="${RUN_NAME:-coverage_hysteresis_b32_t0p90_s75_r25_k3}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
STEPS="${STEPS:-50}"
BUDGET="${BUDGET:-32}"
VIEW_THRESHOLD="${VIEW_THRESHOLD:-0.90}"
GEOMETRIC_WEIGHT="${GEOMETRIC_WEIGHT:-0.75}"
RI_NEIGHBORS="${RI_NEIGHBORS:-3}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR" "$OUTPUT_DIR/profiles" \
  "$METRICS_ROOT" "$VBENCH_OUTPUT_ROOT"
cd "$MEMCAM_ROOT"

echo "MemCam 60s Coverage-Hysteresis Geometric-RI"
echo "Run: $RUN_NAME"
echo "Budget: $BUDGET total stored frames, including the rolling section anchor"
echo "Rows: $ROWS"
echo "View threshold: $VIEW_THRESHOLD"
echo "Geometric Coverage weight: $GEOMETRIC_WEIGHT"
echo "RI weight: 1 - $GEOMETRIC_WEIGHT"
echo "RI rarity neighbors: $RI_NEIGHBORS"
echo "Manifest: $MANIFEST"
echo "Output: $OUTPUT_DIR"
echo "Git commit: $(git rev-parse HEAD)"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"
echo "=== NVIDIA device ==="
nvidia-smi

python -m unittest discover -s tests -p 'test_coverage_hysteresis.py' -v
python -m unittest discover -s tests -p 'test_slam_ri_blend.py' -v
python utils/check_memory_policies.py
python -c 'from diffsynth import ModelManager, save_video; from diffsynth.pipelines.wan_video_memcam import WanVideoMemCamPipeline; print("Wan/MemCam imports passed")'

python utils/run_context_memory_batch.py \
  --manifest "$MANIFEST" \
  --gpu 0 \
  --durations 60 \
  --rows "$ROWS" \
  --output_dir "$OUTPUT_DIR" \
  --num_inference_steps "$STEPS" \
  --memory_policy coverage_hysteresis \
  --memory_budget "$BUDGET" \
  --memory_bank_device cpu \
  --hysteresis_view_threshold "$VIEW_THRESHOLD" \
  --hysteresis_slam_weight "$GEOMETRIC_WEIGHT" \
  --hysteresis_rarity_neighbors "$RI_NEIGHBORS" \
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

echo "=== VBench ($(date)) ==="
conda activate vbench
cd "$VBENCH_ROOT"

python evaluate.py \
  --dimension $VBENCH_DIMENSIONS \
  --videos_path "$OUTPUT_DIR/" \
  --mode custom_input \
  --output_path "$VBENCH_OUTPUT_ROOT/$RUN_NAME"

echo "Finished: $(date)"
echo "Videos: $OUTPUT_DIR"
echo "Summary: $METRICS_ROOT/$RUN_NAME/summary.json"
echo "VBench: $VBENCH_OUTPUT_ROOT/$RUN_NAME"
cat "$METRICS_ROOT/$RUN_NAME/summary.json"
