# MemCam Multi-Cluster Runbook

Last updated: 2026-08-22

For the full research, workflow, evidence, and new-session context, read
[`project_handoff_context.md`](project_handoff_context.md) first. This file is
the infrastructure-specific companion.

This is the operational record for running MemCam across Newton, the CECSL
workstation, Purdue Anvil, and Indiana University Jetstream2. Update the
inventory and run ledger whenever access details or assignments change.

## Rules

1. Use the same Git commit, model weights, manifest split, inference steps,
   seed, resolution, and policy configuration for every shard.
2. Assign every manifest row to exactly one cluster. Do not run a complete
   job and sharded jobs for the same output directory at the same time.
3. Treat generated videos as immutable. Existing outputs are skipped by
   `utils/run_context_memory_batch.py`; do not use `--overwrite` for recovery.
4. Merge all quality-evaluation inputs into one canonical archive before
   computing FVD. FVD is a distribution metric and must use one pooled set.
5. Run every point in a latency/VRAM comparison on the same GPU model and
   software stack. Do not mix A100, H100, and virtual-GPU timings in one curve.
6. Never commit passwords, private keys, ACCESS tokens, OpenStack credentials,
   or Hugging Face tokens.

## Cluster Inventory

| Cluster | Execution model | GPU path | Repository | Output location | State |
| --- | --- | --- | --- | --- | --- |
| Newton | Slurm | H100 PCIe | `~/MemCam` | `~/memcam_results` | Active, queue can be slow |
| CECSL | Direct workstation with `tmux` | Local GPU | `~/MemCam` | `/data/ab575577/MemCam/outputs` | Active |
| Anvil | Slurm | `gpu` for A100, `ai` for H100 | `$PROJECT/$USER/MemCam` | `$SCRATCH/MemCam/outputs` while running | Setup required |
| Jetstream2 | Persistent OpenStack VM with `tmux` | Full A100 or H100 flavor | Attached volume | Attached volume | Setup required |

The proposed canonical archive is:

```bash
/data/ab575577/MemCam/outputs
```

Cluster-specific values to fill in after the first login:

```text
ANVIL_USER=
ANVIL_ACCOUNT=
ANVIL_PROJECT=
ANVIL_SCRATCH=

JETSTREAM2_ALLOCATION=
JETSTREAM2_INSTANCE_NAME=
JETSTREAM2_GPU_FLAVOR=
JETSTREAM2_PUBLIC_IP=
JETSTREAM2_VOLUME_NAME=
JETSTREAM2_VOLUME_MOUNT=
ALLOCATION_EXPIRATION_DATE=
```

Verify the exact expiration date in ACCESS instead of relying on the current
"roughly two months" estimate.

## Reproducibility Record

Before starting a new shard, record:

```bash
date
hostname
git rev-parse HEAD
git status --short
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

The current trajectory-coverage implementation was tested locally at commit:

```text
9e476bbb36ed
```

All clusters should run that commit or a later explicitly recorded commit.

## Manifest Portability

MemCam manifests contain absolute dataset paths. Each cluster therefore needs
its own regenerated manifest, but it must describe the same deterministic
split and preserve the same `output_prefix` values.

For the existing 60-second benchmark, regenerate the full default manifest,
not a new 60-second-only split:

```bash
cd ~/MemCam

python utils/create_context_memory_testbed.py \
  --dataset_root /ABSOLUTE/PATH/TO/Context-as-Memory-Dataset \
  --output_dir testbeds/context_memory \
  --seeds 0 \
  --scenes_per_split 15 \
  --durations 10,20,40,60,120
```

Because the manifest is scene-major, the 15 rows for the 60-second duration
are:

```text
3,8,13,18,23,28,33,38,43,48,53,58,63,68,73
```

Do not use `--rows 0-14` with this manifest; those are not the 15 60-second
runs.

Suggested four-way shard:

| Shard | Rows |
| --- | --- |
| A | `3,8,13,18` |
| B | `23,28,33,38` |
| C | `43,48,53,58` |
| D | `63,68,73` |

Assign the shard letters to clusters in the run ledger only after each new
cluster passes a one-row smoke test.

## Portable Trajectory-Coverage Command

Use the same command on a GPU compute node or inside a Jetstream2/CECSL
`tmux` session. Change only `MANIFEST`, `OUTPUT_DIR`, and `ROWS`.

```bash
cd /PATH/TO/MemCam
conda activate memcam

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/PERSISTENT/PATH/hf_cache

MANIFEST=/PATH/TO/testbeds/context_memory/manifest.jsonl
OUTPUT_DIR=/PATH/TO/outputs/context_memory_60s/trajectory_b32_coverage
ROWS=3,8,13,18

nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

python utils/run_context_memory_batch.py \
  --manifest "$MANIFEST" \
  --rows "$ROWS" \
  --durations 60 \
  --gpu 0 \
  --output_dir "$OUTPUT_DIR" \
  --num_inference_steps 50 \
  --memory_policy trajectory_coverage \
  --memory_budget 32 \
  --memory_bank_device cpu
```

Check completed videos without opening a pager:

```bash
find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*_60s_custom.mp4' | sort
find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*_60s_custom.mp4' | wc -l
tail -n 20 "$OUTPUT_DIR/run_status.jsonl"
```

## Newton

Newton uses the repository's `slurm/newton_*.sbatch` jobs.

```bash
cd ~/MemCam
mkdir -p logs
sbatch --exclude=evc33,evc40 \
  slurm/newton_memcam_h100_60s_trajectory_coverage.sbatch
```

Monitor:

```bash
squeue --me
sacct -j JOB_ID -X --format=JobID,State,ExitCode,Elapsed,NodeList,Reason
tail -n 80 logs/memcam_60s_traj_JOB_ID.out
tail -n 80 logs/memcam_60s_traj_JOB_ID.err
```

Known Newton outcomes:

| Date | Job | Node | Result |
| --- | --- | --- | --- |
| 2026-07-29 | `722162` | `evc33` | Failed before inference: no CUDA GPUs available |
| 2026-07-29 | `722181` | `evc40` | Failed before inference: CUDA unknown error |
| 2026-07-30 | `722256` | `evc44` | Completed in `16:00:55`: 15 videos and 15 access traces |

The two failures occurred in CUDA initialization before trajectory coverage
ran. Job `722256` then completed the full experiment successfully on `evc44`.

## CECSL

Use the direct machine for generation, metric computation, result merging, and
plotting. Long jobs must run inside `tmux`.

```bash
ssh ab575577@CECSL4622128797
tmux new -s memcam

cd ~/MemCam
conda activate memcam
nvidia-smi
```

Detach with `Ctrl-b`, then `d`. Reconnect with:

```bash
tmux ls
tmux attach -t memcam
```

Current 180-second output root:

```bash
/data/ab575577/MemCam/outputs/context_180s
```

## Purdue Anvil

Anvil uses Slurm. SSH authentication is key-only, and the Anvil username
normally begins with `x-`.

Login and discover the allocation:

```bash
ssh ANVIL_USER@anvil.rcac.purdue.edu
mybalance
showpartitions
sfeatures
echo "$PROJECT"
echo "$SCRATCH"
```

Important storage:

- `$HOME` is small and should not contain models or Conda environments.
- `$PROJECT` is persistent allocation storage. Put the repository, environment,
  models, dataset, and Hugging Face cache here.
- `$SCRATCH` is for active output and temporary files. Files not accessed for
  30 days are automatically purged without warning.

Initial layout:

```bash
mkdir -p "$PROJECT/$USER"/{envs,models,data,hf_cache}
mkdir -p "$SCRATCH/MemCam"/{outputs,logs}

cd "$PROJECT/$USER"
git clone https://github.com/Metalicana/MemCam.git
cd MemCam
```

Inspect the available GPU software stack before creating the environment:

```bash
module --force purge
module load modtree/gpu
module spider conda
module spider cuda
```

The repository installation baseline is Python 3.10, PyTorch 2.4.0 with
CUDA 12.4 wheels, `requirements.txt`, and `pip install -e .`. Install the
environment under `$PROJECT`, not `$HOME`.

Anvil queue mapping:

- `gpu`: A100 nodes, up to 48 hours.
- `ai`: H100 nodes, up to 48 hours.
- `gpu-debug`: short GPU smoke tests, up to 30 minutes.

An Anvil job must specify both the allocation account and partition. A minimal
H100 header is:

```bash
#!/bin/bash
#SBATCH -A ANVIL_ACCOUNT
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --job-name=memcam_traj
#SBATCH --output=logs/memcam_traj_%j.out
#SBATCH --error=logs/memcam_traj_%j.err
```

Do not submit a Newton `.sbatch` unchanged on Anvil. Newton's partition and
GPU resource names are cluster-specific.

Useful Anvil commands:

```bash
squeue -u "$USER"
wait_time -j JOB_ID
jobinfo JOB_ID
jobscript JOB_ID
seff JOB_ID
jobsu JOB_ID
scancel JOB_ID
```

## Indiana Jetstream2

Jetstream2 is an OpenStack cloud, not a Slurm cluster. Create a GPU instance,
SSH into it, and run MemCam inside `tmux`.

Before provisioning, verify that the allocation includes the separate
Jetstream2 GPU resource. GPU access is not implied by a Jetstream2 CPU
allocation.

Recommended flavors:

- `g3.xl`: one full A100 with 40 GB VRAM.
- `g5.xl`: one full H100 with 80 GB VRAM, when H100 access has been approved.
- Avoid partial A100 flavors for latency/VRAM profiling. Their vGPU setup has
  known profiling and CUDA-debugging limitations.

Create the instance through Exosphere:

1. Select the correct allocation.
2. Choose a current Ubuntu image.
3. Choose a full-GPU flavor.
4. Assign a public IP.
5. Attach a persistent volume large enough for the dataset, models, cache, and
   outputs.
6. Record the instance, flavor, IP, volume, and allocation expiration above.

SSH to an Exosphere-created instance:

```bash
ssh exouser@JETSTREAM2_PUBLIC_IP
```

Verify the GPU before installing MemCam:

```bash
nvidia-smi
df -h
```

Exosphere normally mounts an attached volume under:

```text
/media/volume/VOLUME_NAME
```

Keep the repository, Conda environment, models, dataset, cache, and outputs on
that volume. The default instance root disk is too small for the complete
experiment stack.

Suggested layout:

```bash
VOLUME=/media/volume/VOLUME_NAME
mkdir -p "$VOLUME"/{envs,data,hf_cache,outputs}

cd "$VOLUME"
git clone https://github.com/Metalicana/MemCam.git
cd MemCam
```

Install the base tools once:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs ffmpeg tmux rsync build-essential
```

Run generation:

```bash
tmux new -s memcam
cd /media/volume/VOLUME_NAME/MemCam
conda activate /media/volume/VOLUME_NAME/envs/memcam
nvidia-smi
```

Detach with `Ctrl-b`, then `d`, and monitor with:

```bash
tmux ls
tmux attach -t memcam
nvidia-smi
```

Shelve the instance whenever it will be idle for more than a short break.
Shelving stops running processes, preserves the instance disk, and stops
compute SU consumption. Volumes are persistent but are not automatically
backed up, so copy final results to the canonical archive.

## Cross-Cluster Validation

Before assigning many rows to a new cluster:

1. Run one identical short row on the established machine and the new cluster.
2. Confirm the output frame count, resolution, configuration, and absence of
   NaNs or corrupt video.
3. Record both GPU models, CUDA/PyTorch versions, and Git commit.
4. Check whether visual metrics differ materially.
5. Only then allocate full shards.

Small numerical differences across hardware are possible. Final headline
quality comparisons should ideally be regenerated on one hardware family if
the cross-cluster check shows meaningful drift.

## Result Transfer

Stage incoming results by cluster instead of copying directly over the
canonical run:

```text
/data/ab575577/MemCam/outputs/incoming/anvil/
/data/ab575577/MemCam/outputs/incoming/jetstream2/
```

Examples from CECSL:

```bash
rsync -av --partial \
  ANVIL_USER@anvil.rcac.purdue.edu:ANVIL_OUTPUT_DIR/ \
  /data/ab575577/MemCam/outputs/incoming/anvil/trajectory_b32_coverage/

rsync -av --partial \
  exouser@JETSTREAM2_PUBLIC_IP:JETSTREAM2_OUTPUT_DIR/ \
  /data/ab575577/MemCam/outputs/incoming/jetstream2/trajectory_b32_coverage/
```

After verifying file counts and playable videos, merge only immutable products:

```bash
rsync -av --ignore-existing \
  /data/ab575577/MemCam/outputs/incoming/anvil/trajectory_b32_coverage/ \
  /data/ab575577/MemCam/outputs/context_memory_60s/trajectory_b32_coverage/

rsync -av --ignore-existing \
  /data/ab575577/MemCam/outputs/incoming/jetstream2/trajectory_b32_coverage/ \
  /data/ab575577/MemCam/outputs/context_memory_60s/trajectory_b32_coverage/
```

Keep each cluster's `run_status.jsonl` and logs separately for provenance.

## Run Ledger

Update this table at submission and completion.

| Date | Experiment | Commit | Cluster | GPU | Rows | Job/session | Status | Output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-29 | trajectory coverage, B=32, 60s | `9e476bbb36ed` | Newton | H100 | all | `722162` | CUDA failure on `evc33` | `~/memcam_results/context_memory_60s/trajectory_b32_coverage` |
| 2026-07-29 | trajectory coverage, B=32, 60s | `9e476bbb36ed` | Newton | H100 | all | `722181` | CUDA failure on `evc40` | same |
| 2026-07-30 | trajectory coverage, B=32, 60s | `9e476bbb36ed` | Newton | H100 | all 15 | `722256` | Completed in `16:00:55`; 15 videos and 15 traces | same |

## Trajectory-Coverage B32 Quality

Evaluation completed on 2026-07-31 with 15 videos at each prefix, LPIPS frame
stride 30, and 60 FVD clips per prefix.

| Prefix | LPIPS Alex | FVD |
| --- | ---: | ---: |
| 10s | 0.493182 | 733.211876 |
| 20s | 0.540887 | 678.929208 |
| 30s | 0.556323 | 696.075294 |
| 60s | 0.591565 | 741.335088 |

Metrics path:

```text
~/memcam_results/eval_prefix_duration_curves_60s_b32/trajectory_b32_coverage/summary.json
```

The `overall` values in that summary are means across four prefixes. Use the
individual `by_duration` values for comparisons and plots.

## Official Documentation

- [Anvil user guide](https://docs.rcac.purdue.edu/userguides/anvil/)
- [Anvil getting started, storage, and allocation commands](https://docs.rcac.purdue.edu/userguides/anvil/getting-started/)
- [Anvil Slurm partitions and GPU job submission](https://docs.rcac.purdue.edu/userguides/anvil/jobs/)
- [Jetstream2 instance flavors](https://docs.jetstream-cloud.org/general/instance-flavors/)
- [Jetstream2 Exosphere instance creation](https://docs.jetstream-cloud.org/ui/exo/create_instance/)
- [Jetstream2 Exosphere SSH access](https://docs.jetstream-cloud.org/ui/exo/access-instance/)
- [Jetstream2 volumes](https://docs.jetstream-cloud.org/general/volume/)
- [Jetstream2 quotas](https://docs.jetstream-cloud.org/general/quotas/)
- [Jetstream2 instance lifecycle and shelving](https://docs.jetstream-cloud.org/general/instancemgt/)
