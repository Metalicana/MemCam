#!/bin/bash
#SBATCH --job-name=keep_ci180
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=keep_ci180_%j.out
#SBATCH --error=keep_ci180_%j.err

set -euo pipefail
module load cuda/cuda-12.1.0
module load ffmpeg
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
PY="${MEMCAM_ENV_PATH:-$HOME/.conda/envs/memcam}/bin/python"
export PATH="$(dirname "$PY"):$PATH"
cd "${MEMCAM_ROOT:-$HOME/MemCam}"
OUT="${OUT:-$HOME/memcam_results/headline_uncertainty_180s_matched15}"
timeout 180s "$PY" -u -c 'import torch; torch.cuda.init(); assert torch.cuda.device_count() == 1; x = torch.ones(1, device="cuda"); x.mul_(2); torch.cuda.synchronize(); assert x.item() == 2; print(torch.cuda.get_device_name(0), flush=True)'
# Eight clips/stride eight reproduce the historical 180s metric definition.
# Both policies now use the same fifteen videos and LPIPS frame stride 90.
"$PY" -u paper/paired_quality_uncertainty.py extract \
  --manifest testbeds/context_memory_180s/manifest.jsonl \
  --baseline "$HOME/memcam_results/context_180s/baseline" \
  --keepsake "$HOME/memcam_results/context_180s/slam_b32_covisibility" \
  --duration 180 --expected-videos 15 --lpips-stride 90 --clips 8 --clip-stride 8 \
  --output "$OUT"
"$PY" -u paper/paired_quality_uncertainty.py report \
  --duration 180 --expected-videos 15 --draws 2000 --output "$OUT" "$@"
