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
cd ~/MemCam/CUT3R
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
```

Expected default path:

```text
~/MemCam/CUT3R/src/cut3r_512_dpt_4_64.pth
```

If you use the 224 checkpoint, pass `CUT3R_SIZE=224` and set `CUT3R_MODEL`.

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
