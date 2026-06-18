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

Run perceptual/semantic metrics for a completed 10s policy run:

```bash
RUN_NAME=ri_b32 \
LEARNED_METRICS=dino,clip \
FRAME_STRIDE=4 \
bash scripts/evaluate_context_memory_10s.sh
```

LPIPS is also supported, but needs the extra package in the active env:

```bash
pip install lpips

RUN_NAME=ri_b32 \
LEARNED_METRICS=lpips,dino,clip \
FRAME_STRIDE=4 \
bash scripts/evaluate_context_memory_10s.sh
```

Metric direction:

- Higher is better: `psnr_db`, `ssim`, `dino_cosine`, `clip_image_cosine`.
- Lower is better: `mae`, `rmse`, `lpips_alex`, `dino_distance`, `clip_image_distance`, temporal delta errors.

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
