#!/bin/bash
#SBATCH --job-name=budget_metrics
#SBATCH --output=budget_metrics_%A_%a.out
#SBATCH --error=budget_metrics_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc33,evc40,evc44
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-83%2

set -euo pipefail
MEMCAM_ENV_PATH="${MEMCAM_ENV_PATH:-$HOME/.conda/envs/memcam}"
MEMCAM_PYTHON="$MEMCAM_ENV_PATH/bin/python"
if [[ ! -x "$MEMCAM_PYTHON" ]]; then
    echo "No Python at $MEMCAM_PYTHON; set MEMCAM_ENV_PATH to the memcam environment directory." >&2
    exit 1
fi
module load cuda
module load ffmpeg
# Evaluator subprocesses use conda run, not interactive shell activation.
if ! type -P conda >/dev/null; then
    module load anaconda
fi
if ! type -P conda >/dev/null; then
    echo "No conda executable on PATH after loading anaconda." >&2
    exit 1
fi
export PATH="$MEMCAM_ENV_PATH/bin:$PATH"
echo "Launcher Python: $MEMCAM_PYTHON"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${TMPDIR:-/tmp}/memcam_mpl_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
MEMCAM_ROOT="${MEMCAM_ROOT:-$HOME/MemCam}"
cd "$MEMCAM_ROOT"
PLAN="${1:-}"
if [[ -z "$PLAN" ]]; then
    OUTPUT="${BUDGET_METRICS_OUTPUT:-$HOME/memcam_results/budget_metrics_60s_${SLURM_ARRAY_JOB_ID:?Missing array job ID}}"
    mkdir -p "$(dirname "$OUTPUT")"
    # Array tasks share one frozen plan; only one prepares it.
    (
        flock -x 9
        if [[ ! -f "$OUTPUT/.prepared" ]]; then
            if [[ -e "$OUTPUT" ]]; then
                echo "Incomplete preparation at $OUTPUT; choose a new BUDGET_METRICS_OUTPUT." >&2
                exit 1
            fi
            ARGS=(prepare --output "$OUTPUT")
            AUDIT="${BUDGET_METRICS_AUDIT:-$HOME/memcam_results/cohort_audit_20260914_135301/audit.json}"
            if [[ -f "$AUDIT" ]]; then
                ARGS+=(--reuse-audit "$AUDIT")
            fi
            "$MEMCAM_PYTHON" utils/run_budget_metric_grid.py "${ARGS[@]}"
            touch "$OUTPUT/.prepared"
        fi
    ) 9>"$OUTPUT.prepare.lock"
    PLAN="$OUTPUT/plan.json"
fi
"$MEMCAM_PYTHON" utils/run_budget_metric_grid.py run \
    --plan "$PLAN" --task "${SLURM_ARRAY_TASK_ID:?Missing array ID}"
