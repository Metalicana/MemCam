#!/bin/bash
#SBATCH --job-name=matched15_metrics
#SBATCH --output=logs/matched15_metrics_%A_%a.out
#SBATCH --error=logs/matched15_metrics_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc33,evc40,evc44
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-7%1

# Four primary policies x two independent evaluators. One GPU at a time.
# Even task IDs: VBench-Long. Odd task IDs: CUT3R (provisional, not calibrated).
# Fresh outputs by default; RESUME=1 resumes the latest matching suite.
# Never resume while the original task is still running.
set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate memcam

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/memcam_results/context_memory_60s}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memcam_results/matched15_metrics_60s}"
VBENCH_ROOT="${VBENCH_ROOT:-$HOME/VBench}"
MANIFEST="${MANIFEST:-$MEMCAM_ROOT/testbeds/context_memory/manifest.jsonl}"

RUNS=(baseline fifo_b32 ri_b32_dino_rgb slam_b32_covisibility)
EVALUATORS=(vbench-long cut3r)
TASK="${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}"
if (( TASK < 0 || TASK > 7 )); then
    echo "Invalid task ID: $TASK" >&2
    exit 2
fi
RUN="${RUNS[$((TASK / 2))]}"
EVALUATOR="${EVALUATORS[$((TASK % 2))]}"
cd "$MEMCAM_ROOT"

echo "Run=$RUN evaluator=$EVALUATOR node=${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
EXTRA_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--resume-latest)
fi
python utils/run_matched_video_metrics.py \
    --run "$RUN" --only "$EVALUATOR" \
    --manifest "$MANIFEST" --results-root "$RESULTS_ROOT" \
    --output-root "$OUTPUT_ROOT" --vbench-root "$VBENCH_ROOT" "${EXTRA_ARGS[@]}"
