#!/bin/bash
#SBATCH --job-name=memcam_ub_audit
#SBATCH --output=logs/memcam_ub_audit_%j.out
#SBATCH --error=logs/memcam_ub_audit_%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

module load anaconda
conda activate memcam

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/memcam_matplotlib_cache_$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/memcam_xdg_cache_$USER}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"
ROOT="${ROOT:-$HOME/memcam_results/context_memory_60s}"
TRACE_DIR="${TRACE_DIR:-$ROOT/baseline/access_traces}"
MAX_ROWS="${MAX_ROWS:-15}"
ROWS="${ROWS:-}"
SECTIONS="${SECTIONS:-auto}"
SECTION_COUNT="${SECTION_COUNT:-5}"
TARGET_STRIDE="${TARGET_STRIDE:-19}"
POOL_SIZES="${POOL_SIZES:-32,128,all}"
REPEATS="${REPEATS:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
DATASET_ROOT="${DATASET_ROOT:-}"

if [ "$MAX_ROWS" = "15" ] && [ -z "$ROWS" ]; then
  OUT_DIR="${OUT_DIR:-$ROOT/unbounded_retrieval_audit}"
else
  OUT_DIR="${OUT_DIR:-$ROOT/unbounded_retrieval_audit_${MAX_ROWS}rows}"
fi

mkdir -p "$MEMCAM_ROOT/logs" "$OUT_DIR"
cd "$MEMCAM_ROOT"

if [ ! -f "$MANIFEST" ]; then
  echo "Missing manifest: $MANIFEST" >&2
  exit 1
fi
if [ ! -d "$TRACE_DIR" ]; then
  echo "Missing unbounded access traces: $TRACE_DIR" >&2
  exit 1
fi

echo "MemCam unbounded retrieval instability audit"
echo "Manifest: $MANIFEST"
echo "Trace dir: $TRACE_DIR"
echo "Output: $OUT_DIR"
echo "Maximum rows: $MAX_ROWS"
echo "Explicit rows: ${ROWS:-none}"
echo "Sections: $SECTIONS"
echo "Section count: $SECTION_COUNT"
echo "Target stride: $TARGET_STRIDE"
echo "Candidate pools: $POOL_SIZES"
echo "Repeats: $REPEATS"
echo "Samples per candidate: $NUM_SAMPLES"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

python -c 'import matplotlib, numpy, torch; print("torch", torch.__version__)'

cmd=(
  python utils/analyze_unbounded_retrieval_instability.py
  --manifest "$MANIFEST"
  --trace_dir "$TRACE_DIR"
  --duration 60
  --max_rows "$MAX_ROWS"
  --sections "$SECTIONS"
  --section_count "$SECTION_COUNT"
  --target_stride "$TARGET_STRIDE"
  --pool_sizes "$POOL_SIZES"
  --repeats "$REPEATS"
  --num_samples "$NUM_SAMPLES"
  --torch_threads 1
  --output_dir "$OUT_DIR"
)

if [ -n "$ROWS" ]; then
  cmd+=(--rows "$ROWS")
fi
if [ -n "$DATASET_ROOT" ]; then
  cmd+=(--dataset_root "$DATASET_ROOT")
fi

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

echo
echo "Finished: $(date)"
echo "Report: $OUT_DIR/report.md"
echo
cat "$OUT_DIR/report.md"
