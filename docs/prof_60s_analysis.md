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
└── memory_mechanisms/
    ├── report.md
    ├── figures/
    │   ├── 01_retrieval_overlap_vs_unbounded.png
    │   ├── 02_preserved_needed_frames_vs_fifo.png
    │   ├── 03_eviction_regret.png
    │   ├── 04_retained_age_distribution.png
    │   └── 05_retained_timeline_row_*.png
    └── tables/
        ├── retrieval_overlap_targets.csv
        ├── retrieval_overlap_summary.csv
        ├── preservation_vs_fifo.csv
        ├── eviction_regret_events.csv
        ├── eviction_regret_summary.csv
        └── retained_age_summary.csv
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

## Mechanism-Only Trace Analysis

If the generated runs already have `access_traces/*.jsonl`, this is the fastest
analysis for the advisor question:

```bash
cd ~/MemCam
ROOT="$HOME/memcam_results/context_memory_60s"
RUNS="baseline,fifo_b32,fifo_b64,ri_b32_dino_rgb,ri_b64_dino_rgb,slam_b16_covisibility,slam_b32_covisibility,slam_b64_covisibility"

python utils/analyze_memory_mechanisms.py \
  --root "$ROOT" \
  --runs "$RUNS" \
  --baseline_run baseline \
  --durations 60 \
  --output_dir "$ROOT/prof_60s_analysis/memory_mechanisms" \
  --near_frame_window 4 \
  --max_timeline_videos 3
```

The three claims it tests are:

- Retrieval overlap: when unbounded retrieves frame X, does a bounded run still
  have X available, and does it actually select X?
- Preservation versus FIFO: how often does the bounded method keep an
  unbounded-needed frame that matched-budget FIFO no longer has?
- Eviction regret: after a policy evicts frame X, does unbounded later retrieve
  X or a nearby frame?

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
- `memory_mechanisms/report.md` for the trace-backed answer to what each policy
  retained, what FIFO discarded, and whether evictions were later regretted.
