# Newton Slurm Notes

## Recommended: H100

Run one 20-step smoke row:

```bash
cd ~/MemCam
sbatch slurm/newton_memcam_h100_smoke.sbatch
```

Run the 75-row benchmark as a Slurm array:

```bash
cd ~/MemCam
sbatch slurm/newton_memcam_h100_array.sbatch
```

Run only the 20s rows on Newton:

```bash
cd ~/MemCam
sbatch slurm/newton_memcam_h100_20s_array.sbatch
```

Run the 20s rows as three separate policy jobs:

```bash
cd ~/MemCam

POLICY=baseline \
MEMORY_POLICY=unbounded \
sbatch slurm/newton_memcam_h100_20s_policy_array.sbatch

POLICY=fifo_b32 \
MEMORY_POLICY=fifo \
MEMORY_BUDGET=32 \
sbatch slurm/newton_memcam_h100_20s_policy_array.sbatch

POLICY=ri_b32_dino_rgb \
MEMORY_POLICY=rarity_irreplaceability \
MEMORY_BUDGET=32 \
sbatch slurm/newton_memcam_h100_20s_policy_array.sbatch
```

On the current cluster, run only the 10s rows:

```bash
cd ~/MemCam
bash scripts/run_current_cluster_10s.sh
```

Evaluate only the 10s rows for a completed policy run:

```bash
cd ~/MemCam
RUN_NAME=baseline bash scripts/evaluate_context_memory_10s.sh
RUN_NAME=fifo_b32 bash scripts/evaluate_context_memory_10s.sh
```

Run the metrics we actually care about for a completed 10s policy run:

```bash
python -m pip install --no-cache-dir lpips

RUN_NAME=ri_b32_dino_rgb \
LEARNED_METRICS=lpips,dino,fvd \
FRAME_STRIDE=4 \
bash scripts/evaluate_context_memory_10s.sh
```

`fvd` now uses the canonical I3D Kinetics-400 TorchScript detector used by
StyleGAN-V/Google FVD-style evaluation. The first run caches the detector at
`~/.cache/memcam/i3d_torchscript.pt`; on clusters without outbound internet,
pre-stage that file and run with:

```bash
FVD_DETECTOR_PATH=/path/to/i3d_torchscript.pt \
RUN_NAME=ri_b32_dino_rgb \
LEARNED_METRICS=fvd \
bash scripts/evaluate_context_memory_10s.sh
```

The compact report metrics are:

- Lower is better: `mse`, `dino_distance`, `lpips_alex`, `fvd`.
- Higher is better: `ssim`.
- The I3D FVD configuration is recorded in `summary.json`, including backend,
  detector path, clip length, frame stride, and resize size.

Copy a tab-separated summary for Excel:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/data/ab575577/MemCam/eval/context_memory")
runs = ["baseline", "fifo_b32", "ri_b32_dino_rgb"]
fields = ["mse", "ssim", "dino_distance", "lpips_alex", "fvd"]
print("run\t" + "\t".join(fields))
for run in runs:
    data = json.loads((root / run / "summary.json").read_text())
    overall = data["overall"]
    print(run + "\t" + "\t".join(
        "" if overall.get(field) is None else f"{overall[field]:.4f}"
        for field in fields
    ))
PY
```

Run offline memory-policy analysis with a Belady oracle:

```bash
cd ~/MemCam

python utils/analyze_memory_policies.py \
  --manifest testbeds/context_memory/manifest.jsonl \
  --dataset_root /data/ab575577/Context-as-Memory-Dataset/Context-as-Memory-Dataset \
  --durations 10 \
  --budgets 32 \
  --policies unbounded,fifo,ri,belady,coverage_oracle
```

Main outputs:

```bash
/data/ab575577/MemCam/analysis/context_memory/policy_aggregate.csv
/data/ab575577/MemCam/analysis/context_memory/policy_summary.csv
/data/ab575577/MemCam/analysis/context_memory/policy_traces.jsonl
/data/ab575577/MemCam/analysis/context_memory/frame_usefulness.csv
/data/ab575577/MemCam/analysis/context_memory/ri_frame_scores.jsonl
```

Inspect the highest ground-truth-usefulness frames:

```bash
python - <<'PY'
import csv
from pathlib import Path

p = Path("/data/ab575577/MemCam/analysis/context_memory/frame_usefulness.csv")
rows = sorted(csv.DictReader(p.open()), key=lambda r: int(r["future_use_count"]), reverse=True)
for r in rows[:20]:
    print(r["row"], r["scene"], r["frame_idx"], r["global_frame_idx"], r["future_use_count"], r["next_use_distance"])
PY
```

Inspect RI score versus ground-truth future usefulness at eviction points:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("/data/ab575577/MemCam/analysis/context_memory/ri_frame_scores.jsonl")
rows = [json.loads(line) for line in p.open()]
rows = sorted(rows, key=lambda r: (r["row"], r["section_idx"], -(r["gt_future_use_count"] or 0)))
for r in rows[:30]:
    print(
        "row", r["row"], "sec", r["section_idx"], "f", r["frame_idx"],
        "gt", r["gt_future_use_count"], "ri", round(r["ri_score"], 4),
        "ri_rank", r["ri_rank"], "gt_rank", r["gt_future_rank"],
        "kept", r["kept_after"], "evicted", r["evicted"],
    )
PY
```

Summarize RI score alignment against ground-truth usefulness:

```bash
python utils/summarize_ri_alignment.py \
  --scores /data/ab575577/MemCam/analysis/context_memory/ri_frame_scores.jsonl \
  --output_dir /data/ab575577/MemCam/analysis/context_memory \
  --topk 32
```

Main outputs:

```bash
/data/ab575577/MemCam/analysis/context_memory/ri_alignment_summary.json
/data/ab575577/MemCam/analysis/context_memory/ri_alignment_by_decision.csv
```

Use separate output folders per memory policy:

```bash
POLICY=baseline MEMORY_POLICY=unbounded bash scripts/run_current_cluster_10s.sh
POLICY=fifo_b32 MEMORY_POLICY=fifo MEMORY_BUDGET=32 bash scripts/run_current_cluster_10s.sh
```

The batch runner writes memory access traces under each run's output directory:

```bash
/data/ab575577/MemCam/outputs/context_memory/<POLICY>/access_traces/*.jsonl
```

Summarize those traces:

```bash
python utils/summarize_access_traces.py \
  --trace_dir /data/ab575577/MemCam/outputs/context_memory/ri_b32/access_traces \
  --output_dir /data/ab575577/MemCam/analysis/context_memory/ri_b32_access
```

Main outputs:

```bash
access_summary.csv
access_summary.json
access_selected_frames.csv
```

Analyze which generated frames were actually useful in the 60s trace runs:

```bash
cd ~/MemCam

python utils/analyze_trace_usefulness.py \
  --root "$HOME/memcam_results/context_memory_60s" \
  --baseline_run baseline \
  --runs baseline,fifo_b64,ri_b64_dino_rgb,fifo_b128,ri_b128_dino_rgb \
  --durations 60 \
  --require_common_targets \
  --output_dir "$HOME/memcam_results/context_memory_60s/trace_usefulness_analysis"
```

This treats the unbounded `baseline` run as the trace upperbound: for each
target context slot, it records the frame that all-memory retrieval selected,
then measures how often each bounded policy selected the same or nearby frame
and how much camera-overlap quality it retained. Main outputs:

```bash
trace_usefulness_report.md
trace_run_summary.csv
trace_section_summary.csv
trace_target_alignment.csv
trace_upperbound_useful_frames.csv
trace_policy_selected_frames.csv
trace_frame_alignment.csv
trace_eviction_summary.csv
```

Analyze manifest trajectory diversity:

```bash
python utils/analyze_trajectory_diversity.py \
  --manifest testbeds/context_memory/manifest.jsonl \
  --output_dir /data/ab575577/MemCam/analysis/context_memory \
  --durations 10,20,40,60,120
```

Main outputs:

```bash
/data/ab575577/MemCam/analysis/context_memory/trajectory_diversity.csv
/data/ab575577/MemCam/analysis/context_memory/trajectory_diversity_summary.csv
```

FIFO requires an explicit memory budget:

```bash
POLICY=fifo MEMORY_POLICY=fifo MEMORY_BUDGET=32 bash scripts/run_current_cluster_10s.sh
POLICY=fifo MEMORY_POLICY=fifo MEMORY_BUDGET=32 sbatch slurm/newton_memcam_h100_20s_array.sbatch
```

Rarity x irreplaceability also requires an explicit memory budget. This policy uses
DINOv2 features for rarity clustering and downsampled RGB nearest-neighbor distance
for irreplaceability. Use a fresh run name for this corrected version; older `ri_b32`
runs used a pose-based heuristic.

```bash
POLICY=ri_b32_dino_rgb MEMORY_POLICY=rarity_irreplaceability MEMORY_BUDGET=32 bash scripts/run_current_cluster_10s.sh
POLICY=ri_b32_dino_rgb MEMORY_POLICY=rarity_irreplaceability MEMORY_BUDGET=32 sbatch slurm/newton_memcam_h100_20s_array.sbatch
```

Baseline/unbounded:

```bash
POLICY=baseline MEMORY_POLICY=unbounded bash scripts/run_current_cluster_10s.sh
POLICY=baseline MEMORY_POLICY=unbounded sbatch slurm/newton_memcam_h100_20s_array.sbatch
```

The array script defaults to rows `0-74` with at most 8 concurrent jobs:

```bash
#SBATCH --array=0-74%8
```

Override paths or steps without editing the script:

```bash
MEMCAM_ROOT=/path/to/MemCam \
MANIFEST=/path/to/MemCam/testbeds/context_memory/manifest.jsonl \
OUTPUT_DIR=/scratch/$USER/MemCam/outputs/context_memory \
STEPS=50 \
sbatch slurm/newton_memcam_h100_array.sbatch
```

## V100 Warning

The current inference path hardcodes `torch.bfloat16`. V100 GPUs do not have native BF16 support, so H100/A100/Blackwell/L40-class GPUs are the safe target. V100 may fail or be much slower unless the code is modified and validated for fp16 inference.

## After Moving The Dataset

Regenerate the manifest on Newton so absolute dataset paths point to the Newton filesystem:

```bash
python utils/create_context_memory_testbed.py \
  --dataset_root /path/on/newton/Context-as-Memory-Dataset/Context-as-Memory-Dataset \
  --output_dir testbeds/context_memory \
  --seeds 0 \
  --scenes_per_split 15 \
  --durations 10,20,40,60,120
```

If you keep an old manifest, pass the Newton dataset root during evaluation:

```bash
python utils/evaluate_context_memory.py \
  --manifest testbeds/context_memory/manifest.jsonl \
  --model_output_dir /scratch/$USER/MemCam/outputs/context_memory/fifo_b32 \
  --dataset_root /path/on/newton/Context-as-Memory-Dataset/Context-as-Memory-Dataset \
  --metrics_dir /scratch/$USER/MemCam/eval/context_memory \
  --run_name fifo_b32 \
  --durations 20
```
