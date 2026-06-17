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

Use separate output folders per memory policy:

```bash
POLICY=fifo bash scripts/run_current_cluster_10s.sh
POLICY=fifo sbatch slurm/newton_memcam_h100_20s_array.sbatch
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
