#!/bin/bash
#SBATCH --job-name=memcam_attn30
#SBATCH --output=logs/memcam_attn30_%j.out
#SBATCH --error=logs/memcam_attn30_%j.err
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
TESTBED_DIR="${TESTBED_DIR:-$MEMCAM_ROOT/testbeds/context_memory_attention_30s_10}"
MANIFEST="$TESTBED_DIR/manifest.jsonl"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/context_memory_attention_30s}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/unbounded_attention_audit}"
AUDIT_DIR="$OUTPUT_DIR/attention_audits"
SUMMARY_DIR="$OUTPUT_DIR/attention_summary"
PROFILE_DIR="$OUTPUT_DIR/profiles"
MAX_ITEMS="${MAX_ITEMS:-10}"
STEPS="${STEPS:-50}"
GPU="${GPU:-0}"
PROBE_SECTIONS="${PROBE_SECTIONS:-3,7,11}"
PROBE_STEPS="${PROBE_STEPS:-5,25,45}"
PROBE_LAYERS="${PROBE_LAYERS:-5,15,25}"
QUERY_COUNT="${QUERY_COUNT:-128}"

mkdir -p "$MEMCAM_ROOT/logs" "$HF_HOME" "$TESTBED_DIR" "$OUTPUT_DIR"
cd "$MEMCAM_ROOT"

echo "MemCam attention/intervention pilot"
echo "Scope: $MAX_ITEMS videos x 30 seconds"
echo "Policy: unbounded (CPU memory bank)"
echo "Probes: sections=$PROBE_SECTIONS steps=$PROBE_STEPS layers=$PROBE_LAYERS"
echo "Output: $OUTPUT_DIR"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Started: $(date)"

timeout 180s python -c \
  'import torch; assert torch.cuda.is_available(); print("CUDA:", torch.cuda.get_device_name(0))'

python -m unittest tests.test_memory_attention_audit

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
  --memory_policy unbounded \
  --memory_bank_device cpu \
  --profile_dir "$PROFILE_DIR" \
  --attention_audit_dir "$AUDIT_DIR" \
  --attention_probe_sections "$PROBE_SECTIONS" \
  --attention_probe_steps "$PROBE_STEPS" \
  --attention_probe_layers "$PROBE_LAYERS" \
  --attention_query_count "$QUERY_COUNT"

python utils/summarize_attention_utility_pilot.py \
  --input_dir "$AUDIT_DIR" \
  --output_dir "$SUMMARY_DIR" \
  --expected_videos "$MAX_ITEMS"

echo "Finished: $(date)"
echo "Generated videos: $(find "$OUTPUT_DIR" -maxdepth 1 -name '*_30s_custom.mp4' | wc -l)"
echo "Complete audits: $(grep -l 'attention_audit_complete' "$AUDIT_DIR"/*.jsonl | wc -l)"
echo "Report: $SUMMARY_DIR/report.md"
