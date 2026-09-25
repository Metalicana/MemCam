# VMem on Newton: Operational Handoff

Prepared 2026-09-25 for the Codex session in `vmem`. This is an operating guide,
not confirmation that VMem is installed or has run on Newton. Do not launch the
existing CECSL controller unchanged. No Newton job was submitted for this handoff.

## Request to the VMem Session

Port the existing VMem transfer workflow to one Slurm-allocated H100 on Newton.
Preserve the audited generation, retention and retrieval settings. First make a
dedicated environment and pass a short native/unbounded versus KEEPSAKE-B32 smoke
test through the first eviction. Then measure throughput and peak memory before
choosing a longer run. Supply one foreground, resumable batch workflow with clear
logs and completion checks. Do not assume a full 60-second pair fits four hours
or 80 GB, and do not silently change the experiment to make it fit.

Read your existing `VMEM_TRANSFER_AUDIT.md`, `VMEM_RESULTS.md`,
`VMEM_CONTROLLED_TRANSFER_V3.md`, and memory-ownership/recovery documentation.
The local VMem worktree has ongoing edits, including untracked results scripts;
preserve them and ensure the needed files are actually transferred to Newton.
A clone/pull includes committed and pushed files, not local untracked files.

## Machines and Paths

| Item | Location / status |
| --- | --- |
| Newton login | `ab575577@newton.ist.ucf.edu` |
| Login hosts | `evuser1` / `evuser2`; use for submission, not generation |
| Local VMem checkout | `/Users/metalicana/projects_summer_2026/vmem` |
| Verified VMem origin | `https://github.com/Metalicana/vmem.git` |
| Proposed Newton VMem checkout | `$HOME/vmem`; existence not checked remotely |
| Existing Newton repos | `$HOME/MemCam`, `$HOME/VBench`, `$HOME/diffusion-forcing-transformer` |
| Existing environments | `$HOME/.conda/envs/memcam`, `vbench`, `dfot` |
| Proposed VMem environment | `$HOME/.conda/envs/vmem`; not yet verified |
| HF cache | `$HOME/hf_cache`; also an existing `$HOME/hf_cache_memcam_clean` |
| Existing MemCam results | `$HOME/memcam_results` |
| Proposed VMem result parent | `$HOME/vmem_results`; create separate experiment roots |

Newton logs sometimes resolve `$HOME` to `/lustre/fs1/home/ab575577` rather than
`/home/ab575577`. These have referred to the same files in supplied logs. Verify
with `readlink -f` before diagnosing a path mismatch. Do not copy CECSL's
`/data/...` or `$HOME/miniconda3/envs/...` paths into Newton jobs.

From the Mac:

```bash
ssh ab575577@newton.ist.ucf.edu
```

On Newton, if the repository is absent:

```bash
cd "$HOME"
git clone https://github.com/Metalicana/vmem.git
cd "$HOME/vmem"
git status --short
git rev-parse HEAD
conda env list
```

If it already exists, inspect its worktree before pulling. Never reset or remove
untracked work to force an update. Do not pull into a checkout while it is being
used by a running experiment.

## GPU Requests: Match the Partition

The user's 2026-09-25 inventory showed:

| Partition | GPU request | Nodes observed |
| --- | --- | --- |
| `highgpu` | `--gres=gpu:nvidia_h100_80gb_hbm3:1` | `evc101` through `evc104` |
| `normal` | `--gres=gpu:nvidia_h100_pcie:1` | General compute nodes |

`highgpu` jobs were accepted under this user's account, and an H100 HBM3 allocation
successfully ran the DFoT experiments. A separate MemCam component job on `evc104`
was observed running. This is not a guarantee of current availability or health.
Do not override only `--partition` on a PCIe-specific script: also change its
`--gres` type. The earlier incompatible request failed at submission.

Inspect live state:

```bash
sinfo -p highgpu -N -O NodeList,Gres,GresUsed
scontrol show partition highgpu
squeue --me
```

`mix` does not mean a GPU is free. All 32 highgpu GPUs were allocated in the
user's earlier snapshot despite mixed node states. `PD (Priority)` is queued,
not failed. Start estimates can move.

For a NEW interactive allocation from a login shell:

```bash
srun --partition=highgpu --nodes=1 --ntasks=1 \
  --gres=gpu:nvidia_h100_80gb_hbm3:1 \
  --cpus-per-task=8 --mem=96G --time=04:00:00 \
  --cpu-bind=none --pty bash -l
```

Inside an existing allocated compute shell, run the environment's Python
directly; do not request another allocation or add an unnecessary nested `srun`.
An earlier nested step failed with `CPU binding outside of job step allocation`.
For an intentionally separate step, request only resources in the allocation;
`--cpu-bind=none` disables explicit CPU binding, not allocation limits.
See [Slurm srun](https://slurm.schedmd.com/srun.html).

## VMem-Specific Blockers Found Locally

These refer to the current local files, not an assumed future implementation.

1. **Impossible free-memory threshold.** `scripts/run_vmem_results.py` defaults
   `--min-free-mib` to **85000**. Newton H100s here are approximately 80 GB; the
   earlier PCIe report showed 79.18 GiB. This admission condition cannot pass.
   Replace the workstation-specific default with an explicit, validated Newton
   requirement and reject a requirement above device capacity immediately. Do
   not just lower it and claim the unbounded rollout will fit. Profile both arms.
2. **Physical GPU masking.** The controller defaults `--gpu` to `1` and rewrites
   child `CUDA_VISIBLE_DEVICES` to that value. Under Slurm, preserve the scheduler's
   mask exactly; use logical `cuda:0` in the process. Add an `inherit` mode through
   the controller and its children. Map NVML diagnostics to the allocated device
   safely; a CUDA logical index need not equal an `nvidia-smi` physical index.
3. **Detached controller.** The default `start` action spawns a detached process.
   A batch script must keep the controller in the foreground and propagate its
   exit code. The existing `run` action is the starting point to audit. Do not
   `nohup`, background, or return from the batch shell while work is still active.
4. **Environment paths.** `scripts/run_vmem_results.sh` defaults to CECSL's
   `$HOME/miniconda3/envs/vmem/bin/python`; use `VMEM_PYTHON` for Newton and pass
   the actual VBench interpreter explicitly. Audit defaults in Python too.
5. **Prior locks/checkpoints are machine-specific inputs.** Existing defaults
   expect `outputs/locks/transfer_v3_controlled.json` and CECSL generation state.
   A git clone does not bring those results, weights, images or checkpoints.
   Establish a new Newton root and lock, or explicitly transfer and validate a
   compatible recovery bundle. Never rewrite an old lock to bypass validation.
6. **GPU admission is not GPU allocation.** Slurm owns the allocation. Do not
   wait for every device on a shared node to be idle or kill another user's/job's
   process. Monitor only the allocated GPU and preserve the scheduler boundary.

Slurm can remap devices inside a job's cgroup; its supplied CUDA mask is not a
physical device-selection suggestion. See [Slurm GRES](https://slurm.schedmd.com/gres.html).

## Environment and Weights

Create a separate `vmem` environment. Do not modify the working `memcam`,
`vbench` or `dfot` environments. The current VMem README specifies Python 3.10;
its requirements pin Torch 2.7.0, torchvision 0.22.0, NumPy 1.24.4, and install
`extern/CUT3R/src/croco/models/curope` as an editable native extension.
Inspect the extension build before installation: determine whether Torch must
be installed first and which CUDA toolkit/compiler is required.

Working DFoT on Newton used Python 3.10, Torch 2.7.1+cu128 and torchvision 0.22.1.
That is evidence that the allocation worked, NOT permission to replace VMem's
dependencies with DFoT's. Likewise, loading CUDA 12.1 was part of older MemCam
jobs, not a universal requirement. A CUDA module and a wheel's runtime are
different; native-extension compilation also needs compatible build tooling.

Record, without reflexively purging modules:

```bash
module list
command -v conda
command -v nvcc
nvcc --version
```

Then implement a repeatable setup script, pin any resolved compatibility fixes,
run `python -m pip check`, and record `pip freeze`, Torch/CUDA versions, module
list and commit. Use the environment's absolute Python path in jobs; `(base)`
in the shell prompt does not matter when that interpreter is invoked explicitly.
Ensure its `bin` directory is on PATH for subprocess tools such as FFmpeg.

The local VMem README requires Hugging Face authentication and access approval
for `liguang0115/vmem`. Confirm access and all actual checkpoint dependencies
before a long allocation. Authenticate interactively; do not put tokens in
scripts, command logs, source control or the handoff. Cache downloads persistently.
Do not download an entire dataset if the existing VMem pilot uses supplied images
and commanded cameras; first inspect its frozen manifest and required assets.

## Batch Skeleton: CUDA Smoke Only

Have the VMem session save/adapt this as `slurm/newton_vmem_gpu_smoke.sbatch`.
It assumes the environment already exists. It intentionally does not launch
generation: the controller fixes above and a real model smoke come first.

```bash
#!/bin/bash
#SBATCH --job-name=vmem_smoke
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --output=vmem_smoke_%j.out
#SBATCH --error=vmem_smoke_%j.err

set -euo pipefail
cd "${VMEM_ROOT:-$HOME/vmem}"
unset PYTHONHOME PYTHONPATH
PY="${VMEM_PYTHON:-$HOME/.conda/envs/vmem/bin/python}"
export PATH="$(dirname "$PY"):$PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
for name in SLURM_JOB_ID SLURM_JOB_NODELIST SLURM_JOB_GPUS CUDA_VISIBLE_DEVICES; do
    printf '%s=%s\n' "$name" "${!name:-unset}"
done
git rev-parse HEAD
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
test -x "$PY"
timeout 180s "$PY" -u - <<'PY'
import os
import sys
import torch
print("Python:", sys.executable, flush=True)
print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda, flush=True)
print("CUDA mask:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
torch.cuda.init()
assert torch.cuda.device_count() == 1, "Expected exactly one allocated visible GPU"
x = torch.ones((256, 256), device="cuda:0")
y = x @ x
torch.cuda.synchronize()
assert y[0, 0].item() == 256.0
print("CUDA kernel OK:", torch.cuda.get_device_name(0), flush=True)
print("Free / total bytes:", torch.cuda.mem_get_info(), flush=True)
PY
```

`nvidia-smi` diagnostics may show more physical devices than the process can
access; the kernel check uses only its allocated logical device. Do not enable
a silent CPU fallback. Extend this check with actual VMem/CUT3R imports, required
CUDA extensions, image/video I/O and a short generation before the full run.

## Monitoring and Recovery

Submit from the checkout so relative log names have a predictable location.
After the VMem session has created the smoke script:

```bash
cd "$HOME/vmem"
sbatch slurm/newton_vmem_gpu_smoke.sbatch
```

Use the returned numeric job ID in place of `JOB_ID`:

```bash
squeue --me
squeue --start -j JOB_ID
sacct -j JOB_ID -X --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
scontrol show job JOB_ID
tail -n 80 -F vmem_smoke_JOB_ID.out vmem_smoke_JOB_ID.err
```

An empty queue is not evidence of success or failure. Check accounting, logs,
the run status and artifact validation. Stopping `tail -F` with Ctrl-C only
stops log viewing. A submitted batch job does not depend on the SSH connection
remaining open. See [Slurm sbatch](https://slurm.schedmd.com/sbatch.html).

Observed failures to avoid repeating:

- Normal-partition job `848503` on `evc34`: `nvidia-smi` found no devices and
  Torch's CUDA preflight failed. No generation occurred in that attempt.
- Earlier `evc30` jobs failed CUDA initialization even inside an explicit step.
  This does not establish one universal cause. Capture allocation/GPU/environment
  details, fail promptly, and report persistent node-level trouble to
  `arcc-request@ucf.edu`; do not repeatedly change model code to hide it.
- VBench on a full 180-second video exhausted a roughly 80 GB H100 in another
  experiment. Validate memory-bounded metric batching separately and preserve
  the metric's mathematical aggregation; do not silently shorten evaluation.

For the final batch workflow, require atomic resumable checkpoints, one writer
per output root, signal handling that stops its own children, and validation of
completed outputs before skipping them. Preserve completed generation when a
metric fails. Do not launch normal/highgpu duplicates into the same output root.
Reuse source/config-compatible checkpoints only; changed hardware/software must
be recorded, and resumption is not uninterrupted latency evidence.

## Expected Deliverables

1. Newton environment setup with pinned dependencies and passed smoke logs.
2. A Slurm-safe foreground adapter: inherited GPU mask, feasible memory checks,
   Newton paths, clear status, and failure exit codes.
3. A short matched pair reaching eviction. Audit pre-eviction pairing, legal
   retrieval, actual resident payload release and generated-frame count using
   the existing VMem checks. KEEPSAKE is the paper name; retain internal legacy
   `GeoCov` identifiers where changing them would break provenance.
4. Measured seconds/action and peak memory for each arm; estimate the long run
   from those measurements, allowing for growth in unbounded reconstruction.
5. One submission command for the approved full study, plus monitor/download
   commands. Distinguish tests run locally from tests run on the allocated GPU.

Keep VMem's native surfel-based retrieval unchanged across the compared arms,
unless the experiment explicitly declares a reader modification. Document the
bound's scope: a 32-view resident bank does not automatically bound surfels,
metadata, all model state, or total process memory. This transfer is separate
from the DFoT observed-history pilot and its results.

For a portable completed results archive, have the runner package a
`results.zip` inside its actual result directory. From the Mac, replacing
`ACTUAL_RUN_DIRECTORY` with the path printed by that runner:

```bash
scp ab575577@newton.ist.ucf.edu:vmem_results/ACTUAL_RUN_DIRECTORY/results.zip \
  "$HOME/Downloads/vmem_newton_results.zip"
```

No final VMem runtime, success claim or quality result follows from this handoff.
