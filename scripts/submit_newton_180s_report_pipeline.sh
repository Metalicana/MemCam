#!/bin/bash

set -euo pipefail

cd "${MEMCAM_ROOT:-$HOME/MemCam}"
mkdir -p logs

MANIFEST="${MANIFEST:-$HOME/MemCam/testbeds/context_memory_180s/manifest.jsonl}"
ROOT="${ROOT:-$HOME/memcam_results/context_180s}"
export MANIFEST ROOT
export OUTPUT_ROOT="$ROOT"

submit_job() {
  local submission
  submission=$(sbatch --parsable "$@")
  echo "${submission%%;*}"
}

MISSING_TASKS=$(
  python utils/find_missing_180s_recovery_tasks.py \
    --manifest "$MANIFEST" \
    --root "$ROOT"
)
if [ -z "$MISSING_TASKS" ]; then
  echo "No missing generation outputs."
  GEN_JOB=""
  GENERATION_DEPENDENCY=()
else
  GEN_JOB=$(
    submit_job --array="${MISSING_TASKS}%8" \
      slurm/newton_memcam_h100_180s_finish_missing_array.sbatch
  )
  GENERATION_DEPENDENCY=(--dependency="afterok:$GEN_JOB")
fi
QUALITY_JOB=$(
  submit_job "${GENERATION_DEPENDENCY[@]}" \
    slurm/newton_eval_h100_180s_report_array.sbatch
)
CUT3R_JOB=$(
  submit_job "${GENERATION_DEPENDENCY[@]}" \
    slurm/newton_cut3r_h100_180s_report_array.sbatch
)
REVISIT_JOB=$(
  submit_job "${GENERATION_DEPENDENCY[@]}" \
    slurm/newton_revisit_180s_report.sbatch
)
CUT3R_EVAL_JOB=$(
  submit_job --dependency="afterok:$CUT3R_JOB" \
    slurm/newton_cut3r_eval_180s_report.sbatch
)
FINAL_JOB=$(
  submit_job --dependency="afterok:$QUALITY_JOB:$CUT3R_EVAL_JOB:$REVISIT_JOB" \
    slurm/newton_finalize_180s_report.sbatch
)

echo "generation recovery: ${GEN_JOB:-not needed}"
echo "FVD/LPIPS/DINO:     $QUALITY_JOB"
echo "CUT3R reconstruction: $CUT3R_JOB"
echo "CUT3R evaluation:   $CUT3R_EVAL_JOB"
echo "revisit analysis:   $REVISIT_JOB"
echo "final joined report: $FINAL_JOB"
echo
echo "Monitor: squeue -u $USER"
echo "Final report: $HOME/memcam_results/context_180s/report_180s/final/report.md"
