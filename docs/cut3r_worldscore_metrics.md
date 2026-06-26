# CUT3R And WorldScore Metrics

This is the non-overlap-label evaluation path for Context-as-Memory runs.

## What This Measures

The CUT3R path estimates a camera trajectory from each generated video and
compares it against the manifest camera trajectory:

- `rotation_error_deg_*`: relative camera rotation error.
- `translation_error_scale_only_*`: translation error after one scalar scale
  alignment, matching the spirit of WorldScore camera-control scoring.
- `translation_error_sim3_*`: trajectory-shape error after full Sim(3)
  alignment; useful for reconstruction-shape sanity checks.
- `endpoint_*`: end-of-video drift.
- `loop_*`: endpoint distance mismatch, most useful for round-trip trajectories.
- `worldscore_camera_control_score`: WorldScore-style normalization of mean
  rotation and scale-only translation error, on a 0-100 scale.

This is still not the full WorldScore benchmark. It is the immediately runnable
camera-control part using CUT3R instead of DROID-SLAM.

## CUT3R Checkpoint

CUT3R checkpoints are not committed to this repo. Put one of these under
`CUT3R/src/` on Newton:

```bash
cd ~/MemCam/CUT3R/src
gdown 1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD -O cut3r_512_dpt_4_64.pth
```

Expected default path:

```text
~/MemCam/CUT3R/src/cut3r_512_dpt_4_64.pth
```

If you use the 224 checkpoint, pass `CUT3R_SIZE=224` and set `CUT3R_MODEL`.

## CUT3R Environment

Install the CUT3R Python dependencies once inside the `memcam` env:

```bash
cd ~/MemCam
python -m pip install hydra-core omegaconf roma trimesh viser gradio matplotlib tqdm opencv-python scipy einops tensorboard 'pyglet<2' 'huggingface-hub[torch]>=0.22' pillow==10.3.0 h5py accelerate transformers scikit-learn
```

Compile CUT3R's RoPE CUDA extension once on a GPU node. CUT3R may start without
this extension, but CUDA inference can crash in the fallback PyTorch RoPE path.

```bash
cd ~/MemCam/CUT3R/src/croco/models/curope

module load gcc || true
which gcc
which g++
gcc --version
g++ --version

MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="9.0" \
CC="$(which gcc)" CXX="$(which g++)" CUDAHOSTCXX="$(which g++)" \
python setup.py build_ext --inplace
```

If `g++` is missing or compilation fails with `cannot execute 'cc1plus'`, install
a compiler in the conda environment and retry:

```bash
conda install -c conda-forge gcc_linux-64 gxx_linux-64

cd ~/MemCam/CUT3R/src/croco/models/curope
MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="9.0" \
CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
python setup.py build_ext --inplace
```

Verify the extension import from the repo root:

```bash
cd ~/MemCam
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path("CUT3R/src/croco").resolve()))
from models.curope import cuRoPE2D
print("cuRoPE2D OK:", cuRoPE2D)
PY
```

Before running CUT3R, make sure the shell is inside a GPU allocation:

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

If CUDA is not available, start a fresh GPU shell:

```bash
srun -p normal --gres=gpu:nvidia_h100_pcie:1 --time=5:00:00 --cpus-per-task=16 --mem=128G --exclude=evc22 --pty bash
module load cuda
module load ffmpeg
module load anaconda
conda activate memcam
```

## Smoke Test

Run one generated video per run first:

```bash
cd ~/MemCam

ROOT="$HOME/memcam_results/context_memory_60s"
RUNS="baseline,fifo_b64,ri_b64_dino_rgb,slam_b64_covisibility"

python utils/run_cut3r_context_memory.py \
  --manifest testbeds/context_memory/manifest.jsonl \
  --root "$ROOT" \
  --runs "$RUNS" \
  --output_dir "$ROOT/cut3r_pose_recon_smoke" \
  --cut3r_root "$HOME/MemCam/CUT3R" \
  --model_path "$HOME/MemCam/CUT3R/src/cut3r_512_dpt_4_64.pth" \
  --size 512 \
  --device cuda \
  --frame_stride 30 \
  --max_frames 120 \
  --durations 60 \
  --limit 1

python utils/evaluate_cut3r_camera_metrics.py \
  --cut3r_dir "$ROOT/cut3r_pose_recon_smoke" \
  --output_dir "$ROOT/cut3r_camera_metrics_smoke" \
  --runs "$RUNS" \
  --durations 60
```

## Full 60s Run

Use the Slurm wrapper:

```bash
cd ~/MemCam
sbatch slurm/newton_cut3r_context_memory_60s.sbatch
```

To run only a subset:

```bash
RUNS="baseline,fifo_b64,ri_b64_dino_rgb" \
LIMIT=3 \
sbatch slurm/newton_cut3r_context_memory_60s.sbatch
```

The outputs are:

```text
$HOME/memcam_results/context_memory_60s/cut3r_pose_recon/
$HOME/memcam_results/context_memory_60s/cut3r_camera_metrics/cut3r_camera_summary.csv
$HOME/memcam_results/context_memory_60s/cut3r_camera_metrics/cut3r_camera_metrics.csv
```

## WorldScore

WorldScore is heavier than CUT3R. It expects its own benchmark instance layout,
plus separate dependencies such as DROID-SLAM, GroundingDINO/SAM, SAM2,
VFIMamba, and its checkpoints.

This repo now includes a `memcam_context` WorldScore config and a converter that
maps MemCam generated videos into WorldScore static-evaluation instances.

Prepare a smoke subset:

```bash
cd ~/MemCam

ROOT="$HOME/memcam_results/context_memory_60s"
WS_RUNS_ROOT="$ROOT/worldscore_memcam"
RUNS="baseline,fifo_b64,ri_b64_dino_rgb,slam_b64_covisibility"

python utils/prepare_worldscore_context_memory.py \
  --manifest testbeds/context_memory/manifest.jsonl \
  --root "$ROOT" \
  --runs "$RUNS" \
  --worldscore_runs_root "$WS_RUNS_ROOT" \
  --durations 60 \
  --frame_stride 30 \
  --max_frames 50 \
  --limit 1 \
  --overwrite
```

Then in the WorldScore environment:

```bash
cd ~/MemCam/WorldScore
export WORLDSCORE_PATH="$HOME/MemCam"
export MEMCAM_WORLDSCORE_RUNS_ROOT="$HOME/memcam_results/context_memory_60s/worldscore_memcam"

python worldscore/run_evaluate.py --model_name memcam_context --num_jobs 1
```

Full WorldScore is much more expensive than the CUT3R metric and needs its
third-party checkpoints. Start with CUT3R camera metrics first; use WorldScore
once the adapter smoke subset passes.
