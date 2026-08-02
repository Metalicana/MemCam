#!/bin/bash
#SBATCH --job-name=memcam_sf30
#SBATCH --output=logs/memcam_sf30_%j.out
#SBATCH --error=logs/memcam_sf30_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclude=evc33,evc40,evc44

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

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
BASE_MANIFEST="${BASE_MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
TESTBED_DIR="${TESTBED_DIR:-$MEMCAM_ROOT/testbeds/context_memory_surprise_30s_10}"
MANIFEST="$TESTBED_DIR/manifest.jsonl"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/context_memory_surprise_30s}"
BUDGET="${BUDGET:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/surprise_b${BUDGET}_value}"
TRACE_DIR="$OUTPUT_DIR/access_traces"
PROFILE_DIR="$OUTPUT_DIR/profiles"
SUMMARY_DIR="$OUTPUT_DIR/surprise_summary"
MAX_ITEMS="${MAX_ITEMS:-10}"
STEPS="${STEPS:-50}"
GPU="${GPU:-0}"
VALUE_LAYER="${VALUE_LAYER:-15}"

mkdir -p "$MEMCAM_ROOT/logs" "$HF_HOME" "$TESTBED_DIR" "$OUTPUT_DIR"
cd "$MEMCAM_ROOT"

echo "MemCam Surprise Forcing memory baseline"
echo "Scope: $MAX_ITEMS videos x 30 seconds"
echo "Policy: surprise_forcing"
echo "Total external-memory budget: $BUDGET frames (including sink frame)"
echo "Value-token layer: $VALUE_LAYER"
echo "Output: $OUTPUT_DIR"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Started: $(date)"

timeout 180s python -c \
  'import torch; assert torch.cuda.is_available(); print("CUDA:", torch.cuda.get_device_name(0))'

python -m unittest tests.test_surprise_forcing tests.test_memory_attention_audit

DATASET_ROOT=$(BASE_MANIFEST="$BASE_MANIFEST" python -c \
  'import json, os; from pathlib import Path; rows=(json.loads(line) for line in Path(os.environ["BASE_MANIFEST"]).read_text().splitlines() if line.strip()); print(Path(next(row["gt_frames_dir"] for row in rows if row.get("gt_frames_dir"))).resolve().parents[1])')

echo "Dataset: $DATASET_ROOT"
python utils/create_context_memory_testbed.py \
  --dataset_root "$DATASET_ROOT" \
  --output_dir "$TESTBED_DIR" \
  --seeds 0 \
  --scenes_per_split "$MAX_ITEMS" \
  --durations 30

python utils/run_context_memory_batch.py \
  --manifest "$MANIFEST" \
  --gpu "$GPU" \
  --durations 30 \
  --max_items "$MAX_ITEMS" \
  --output_dir "$OUTPUT_DIR" \
  --num_inference_steps "$STEPS" \
  --memory_policy surprise_forcing \
  --memory_budget "$BUDGET" \
  --memory_bank_device cpu \
  --surprise_alpha 0.7 \
  --surprise_ema_momentum 0.95 \
  --surprise_controller_step 0.1 \
  --surprise_target_admission_ratio 0.3 \
  --surprise_initial_threshold 0.002 \
  --surprise_surprise_weight 1.8 \
  --surprise_usage_weight 1.0 \
  --surprise_age_weight 0.4 \
  --surprise_route_top_k 3 \
  --surprise_value_layer "$VALUE_LAYER" \
  --surprise_warmup_sections 3 \
  --access_trace_dir "$TRACE_DIR" \
  --profile_dir "$PROFILE_DIR"

python utils/summarize_surprise_forcing.py \
  --input_dir "$TRACE_DIR" \
  --output_dir "$SUMMARY_DIR" \
  --expected_videos "$MAX_ITEMS"

echo "Finished: $(date)"
echo "Generated videos: $(find "$OUTPUT_DIR" -maxdepth 1 -name '*_30s_custom.mp4' | wc -l)"
echo "Controller report: $SUMMARY_DIR/report.md"
