#!/bin/bash
#SBATCH --job-name=memcam_revisit_180s
#SBATCH --output=logs/memcam_revisit_180s_%j.out
#SBATCH --error=logs/memcam_revisit_180s_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail

module load ffmpeg
module load anaconda
conda activate memcam

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory_180s/manifest.jsonl}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/memcam_results/context_180s}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/memcam_results/context_180s/revisit_consistency}"
RUNS="${RUNS:-fifo_b32,ri_b32_dino_rgb,slam_b32_covisibility}"
DATASET_ROOT="${DATASET_ROOT:-}"
DURATIONS="${DURATIONS:-180}"
ROWS="${ROWS:-}"
LIMIT="${LIMIT:-}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-30}"
MIN_REVISIT_GAP="${MIN_REVISIT_GAP:-150}"
EXACT_POSITION_THRESHOLD="${EXACT_POSITION_THRESHOLD:-0.25}"
EXACT_ROTATION_DEG="${EXACT_ROTATION_DEG:-5.0}"
FORWARD_AXIS="${FORWARD_AXIS:-x}"
RAY_DISTANCE_THRESHOLD="${RAY_DISTANCE_THRESHOLD:-0.75}"
POINT_CLUSTER_RADIUS="${POINT_CLUSTER_RADIUS:-1.5}"
MIN_DEPTH="${MIN_DEPTH:-1.0}"
MAX_DEPTH="${MAX_DEPTH:-80.0}"
MIN_RAY_ANGLE_DEG="${MIN_RAY_ANGLE_DEG:-5.0}"
PATCH_SIZE="${PATCH_SIZE:-96}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_DIR"
cd "$MEMCAM_ROOT"

echo "MemCam 180s revisit consistency"
echo "Manifest: $MANIFEST"
echo "Model root: $MODEL_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Runs: $RUNS"
echo "Durations: $DURATIONS"
echo "Rows: ${ROWS:-all}"
echo "Sample stride: $SAMPLE_STRIDE"
echo "Min revisit gap: $MIN_REVISIT_GAP"
echo "Forward axis: $FORWARD_AXIS"
echo "Started: $(date)"

cmd=(
  python utils/analyze_revisit_consistency.py
  --manifest "$MANIFEST"
  --model_root "$MODEL_ROOT"
  --runs "$RUNS"
  --output_dir "$OUTPUT_DIR"
  --durations "$DURATIONS"
  --sample_stride "$SAMPLE_STRIDE"
  --min_revisit_gap "$MIN_REVISIT_GAP"
  --exact_position_threshold "$EXACT_POSITION_THRESHOLD"
  --exact_rotation_deg "$EXACT_ROTATION_DEG"
  --forward_axis "$FORWARD_AXIS"
  --ray_distance_threshold "$RAY_DISTANCE_THRESHOLD"
  --point_cluster_radius "$POINT_CLUSTER_RADIUS"
  --min_depth "$MIN_DEPTH"
  --max_depth "$MAX_DEPTH"
  --min_ray_angle_deg "$MIN_RAY_ANGLE_DEG"
  --patch_size "$PATCH_SIZE"
)

if [ -n "$DATASET_ROOT" ]; then
  cmd+=(--dataset_root "$DATASET_ROOT")
fi
if [ -n "$ROWS" ]; then
  cmd+=(--rows "$ROWS")
fi
if [ -n "$LIMIT" ]; then
  cmd+=(--limit "$LIMIT")
fi

"${cmd[@]}"

echo "Finished: $(date)"
echo "Summary:"
cat "$OUTPUT_DIR/tables/revisit_summary.csv"
