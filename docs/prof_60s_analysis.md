# Professor 60s Analysis

Goal: produce a folder of figures and tables explaining why bounded 60s memory
policies can beat the unbounded baseline on LPIPS/FVD.

## One-Command Slurm Run

```bash
cd ~/MemCam
sbatch slurm/newton_prof_60s_analysis.sbatch
```

Default output:

```text
$HOME/memcam_results/context_memory_60s/prof_60s_analysis/
├── report.md
├── figures/
│   ├── 01_main_quality_metrics.png
│   ├── 02_percent_improvement_vs_unbounded.png
│   ├── 03_memory_behavior_vs_quality.png
│   ├── 04_lpips_by_time_bucket.png
│   └── 05_biggest_lpips_wins_contact_sheet.png
└── tables/
    ├── main_quality_table.csv
    ├── joined_memory_quality.csv
    ├── lpips_time_buckets.csv
    └── top_lpips_wins.csv
└── bounded_memory_drivers/
    ├── report.md
    ├── figures/
    │   ├── 01_quality_headline.png
    │   ├── 02_reference_age_shift_vs_unbounded.png
    │   ├── 03_stale_reference_rate.png
    │   ├── 04_age_overlap_tradeoff.png
    │   └── 05_lpips_driver_scatter.png
    └── tables/
        ├── run_trace_summary.csv
        ├── paired_target_deltas.csv
        ├── paired_run_summary.csv
        ├── quality_summary.csv
        └── video_driver_quality.csv
```

The default run set excludes incomplete 60s policies:

```text
baseline,fifo_b32,fifo_b64,ri_b32_dino_rgb,ri_b64_dino_rgb,slam_b16_covisibility,slam_b32_covisibility,slam_b64_covisibility
```

## Faster Smoke Run

Use this to make sure plotting and metrics wiring work:

```bash
cd ~/MemCam
RUNS="baseline,fifo_b64,ri_b64_dino_rgb" \
MAX_FRAMES=300 \
FRAME_STRIDE=30 \
FVD_CLIPS_PER_VIDEO=2 \
OUT_DIR="$HOME/memcam_results/context_memory_60s/prof_60s_analysis_smoke" \
sbatch slurm/newton_prof_60s_analysis.sbatch
```

## Reuse Existing Eval Metrics

If `eval_metrics_60s/<run>/summary.json` and `metrics.csv` already exist:

```bash
cd ~/MemCam
RUN_EVAL=0 sbatch slurm/newton_prof_60s_analysis.sbatch
```

## Manual Report Build

If metrics and trace summaries already exist, build only the figure folder:

```bash
cd ~/MemCam
ROOT="$HOME/memcam_results/context_memory_60s"
RUNS="baseline,fifo_b32,fifo_b64,ri_b32_dino_rgb,ri_b64_dino_rgb,slam_b16_covisibility,slam_b32_covisibility,slam_b64_covisibility"

python utils/build_prof_60s_analysis.py \
  --root "$ROOT" \
  --metrics_dir "$ROOT/eval_metrics_60s" \
  --runs "$RUNS" \
  --baseline_run baseline \
  --duration 60 \
  --output_dir "$ROOT/prof_60s_analysis" \
  --manifest testbeds/context_memory/manifest.jsonl
```

## Bounded-Vs-Unbounded Driver Analysis

If the generated runs already have `access_traces/*.jsonl`, this is the fastest
trace-side analysis for the advisor question. If `eval_metrics_60s` is also
present, it joins per-video LPIPS deltas to the trace behavior.

```bash
cd ~/MemCam
ROOT="$HOME/memcam_results/context_memory_60s"
RUNS="baseline,fifo_b32,fifo_b64,ri_b32_dino_rgb,ri_b64_dino_rgb,slam_b16_covisibility,slam_b32_covisibility,slam_b64_covisibility"

python utils/analyze_bounded_memory_drivers.py \
  --root "$ROOT" \
  --metrics_dir "$ROOT/eval_metrics_60s" \
  --runs "$RUNS" \
  --baseline_run baseline \
  --duration 60 \
  --output_dir "$ROOT/prof_60s_analysis/bounded_memory_drivers"
```

This is the advisor-facing mechanism analysis. It treats unbounded as the
comparison baseline, not as an oracle. It tests:

- Did the bounded policy reduce the candidate menu size?
- Did it shift selected references newer or older than unbounded?
- Did it reduce very stale selected references?
- Did it trade geometric overlap for better LPIPS?
- At video level, do LPIPS wins line up with those retrieval shifts?

## Interpretation

Use `report.md` as the talk track. The intended claim is:

> Bounded memory can improve LPIPS/FVD because unbounded retrieval gives the
> generator too many historical candidates. The extra memory may be
> geometrically relevant but visually conflicting or stale. Bounded policies
> act like memory regularization.

Then point to:

- `01_main_quality_metrics.png` for the headline.
- `02_percent_improvement_vs_unbounded.png` for the bounded-vs-unbounded gap.
- `03_memory_behavior_vs_quality.png` for the mechanism.
- `05_biggest_lpips_wins_contact_sheet.png` for examples.
- `bounded_memory_drivers/report.md` for the trace-backed bounded-vs-unbounded
  mechanism story.
