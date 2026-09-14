#!/bin/bash
#SBATCH --job-name=memcam_vbench_grid
#SBATCH --output=logs/memcam_vbench_grid_%j.out
#SBATCH --error=logs/memcam_vbench_grid_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate memcam

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
VBENCH_ROOT="${VBENCH_ROOT:-$HOME/VBench}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/memcam_results/context_memory_60s}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/metrics/vbench_full_grid_60s_n15}"
STAGE_ROOT="${SLURM_TMPDIR:-/tmp}/memcam_vbench_matched15_${SLURM_JOB_ID:-manual}"
RUNS="${RUNS:-baseline,fifo_b16,fifo_b32,fifo_b64,fifo_b128,ri_b16_dino_rgb,ri_b32_dino_rgb,ri_b64_dino_rgb,ri_b128_dino_rgb,slam_b16_covisibility,slam_b32_covisibility,slam_b64_covisibility,slam_b128_covisibility,kcenter_b16,kcenter_b32,kcenter_b64,kcenter_b128,mce_b16_lambda1_pilot,mce_b32_lambda1_pilot,mce_b64_lambda1_pilot,mce_b128_lambda1_pilot}"
DIMENSIONS="subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality"
FORCE="${FORCE:-0}"

mkdir -p "$MEMCAM_ROOT/logs" "$OUTPUT_ROOT" "$STAGE_ROOT"
cd "$MEMCAM_ROOT"
python utils/stage_matched_video_cohort.py \
  --manifest testbeds/context_memory/manifest.jsonl \
  --source-root "$RESULTS_ROOT" --output-root "$STAGE_ROOT" --runs "$RUNS"

conda activate vbench
cd "$VBENCH_ROOT"
IFS=',' read -r -a RUN_ARRAY <<< "$RUNS"
for run in "${RUN_ARRAY[@]}"; do
  out="$OUTPUT_ROOT/$run"
  if [[ "$FORCE" != "1" ]] && compgen -G "$out/*_eval_results.json" > /dev/null; then
    echo "[skip] $run already has an output JSON"
    continue
  fi
  echo "[run] standard VBench: $run"
  python evaluate.py --dimension $DIMENSIONS \
    --videos_path "$STAGE_ROOT/$run" --mode custom_input --output_path "$out"
done
