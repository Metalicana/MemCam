# CECSL Handoff: DFoT and FramePack Comparisons for KEEPSAKE

## Request to the Next Codex Session

Set up and run DFoT and FramePack on the existing Context-as-Memory evaluation
trajectories so we can compare them with MemCam and MemCam + KEEPSAKE. The
dataset exists on both CECSL and Newton. Discover the installed implementations
and checkpoints, transfer the exact evaluation manifest if necessary, implement
the adapters, and provide one resumable command covering generation, validation,
evaluation, and final CSV/LaTeX export.

Default scope: **15 matched trajectories, nominal 180 seconds per video**, matching
the current MemCam headline. This is 30 new videos if both external baselines are
compatible and neither already has usable outputs. Reuse existing MemCam results;
do not regenerate MemCam, repeat its budget sweep, or launch the pose/appearance
ablation as part of this task. Do not automatically launch an additional 60-second
suite. Confirm available GPU time before starting the full generation workload.

This request is for external baseline comparisons. It does not yet request
DFoT + KEEPSAKE or FramePack + KEEPSAKE. Those would require a separate inspection
of whether the models expose a compatible persistent archive and memory reader.

## 1. Identify Which Implementations We Actually Have

**Having the Context-as-Memory dataset is not the same as having its trained
DFoT and FramePack baselines.** In Section 4.2, the Context-as-Memory paper says
its compared methods were implemented on a shared base model and dataset, with
matching training configurations and iterations. Its Section 4.1 describes an
internal base model. Stock public checkpoints are therefore not automatically
reproductions of those comparisons. [Context-as-Memory paper](https://arxiv.org/html/2506.03141v2#S4.SS2)

Inspect local code and checkpoints before deciding the execution path:

| What is installed? | Action |
| --- | --- |
| Actual camera-conditioned CaM baseline implementations and their trained weights | Prefer these; verify provenance, input contract, and inference settings. |
| Official pretrained DFoT | Inspect its pose-conditioned path and implement a CaM dataset adapter. Record the checkpoint's training domain. |
| Stock pretrained FramePack | Check for an actual camera-conditioned extension before claiming trajectory-matched generation. |
| Dataset only, or code without compatible weights | State exactly what is missing. Do not silently train models or fabricate a compatible checkpoint. |

The official DFoT repository exposes `algorithm=dfot_video_pose` and the
`DFoT_RE10K.ckpt` pretrained example. Its demo reads RealEstate10K, not our
Context-as-Memory manifest; replace the dataset interface, not just the output
directory. [Official DFoT code](https://github.com/kwsong0113/diffusion-forcing-transformer)

The inspected stock FramePack `demo_gradio.py` worker accepts an image, prompt,
seed, duration, and sampling controls, but no camera trajectory. It also rounds
duration to latent sections and uses a nontrivial section ordering. Check the
installed variant, including whether it is original FramePack or F1, and verify
final chronological output. [Official FramePack worker](https://github.com/lllyasviel/FramePack/blob/main/demo_gradio.py)

If FramePack has no camera input, offer a clearly identified native image-to-video
comparison using the same starting images and supported prompts. It is not the
same camera-conditioned task. Do not substitute a camera-motion sentence for
numeric pose conditioning or present exact-index LPIPS as matched-view fidelity.
Ask the user to choose this explicitly before spending the full generation budget.
Continue implementing the compatible DFoT path in the meantime.

## 2. Machines and Data

These are previously recorded locations; verify them locally:

```text
CECSL login:       ab575577@CECSL4622128797
MemCam repo:       /home/ab575577/MemCam
Dataset:          /data/ab575577/Context-as-Memory-Dataset/Context-as-Memory-Dataset
MemCam outputs:   /data/ab575577/MemCam/outputs
180s outputs:     /data/ab575577/MemCam/outputs/context_180s
60s outputs:      /data/ab575577/MemCam/outputs/context_memory_60s

Newton login:     ab575577@newton.ist.ucf.edu
Newton repo:      /home/ab575577/MemCam
Newton outputs:   /home/ab575577/memcam_results
Newton 180s:      /home/ab575577/memcam_results/context_180s
```

Start by checking `hostname`, `nvidia-smi`, `conda env list`, available disk space,
and the local repositories/checkpoint caches. Do not assume CECSL has an H100 or
that DFoT/FramePack environments already exist. Keep their dependencies separate
from the working `memcam` and `vbench` environments.

Use CECSL directly, preferably inside `tmux`; it is not a Newton Slurm node.
Invoke the discovered environment's absolute Python executable from launchers.
Avoid nested `conda run` activation loops. Print import/model-load progress;
do not impose the old 90-second timeout on importing Torch. If the user chooses
Newton instead, provide one sbatch entry point using its actual allocation rules.

Suggested study root, separate from all existing outputs:

```text
/data/ab575577/MemCam/outputs/external_baselines_180s/
    inputs/          source and path-remapped manifests
    dfot/            validated videos and per-video receipts
    framepack/       validated videos and per-video receipts
    metrics/         evaluator outputs, split by model and metric
    logs/            generation/evaluation logs
    tables/          coverage.csv, scores.csv, comparison.tex
    provenance.json
```

## 3. Transfer the Exact Cohort, Not Just the Scene Names

The canonical 180-second manifest is:

```text
~/MemCam/testbeds/context_memory_180s/manifest.jsonl
```

It has 15 rows, indexed 0 through 14. It may exist only on the experiment
machines, not in the Git checkout. If the CECSL copy is missing or differs,
copy Newton's manifest into the new study's `inputs/` directory. For example,
run on CECSL after verifying the destination:

```bash
mkdir -p /data/ab575577/MemCam/outputs/external_baselines_180s/inputs
scp ab575577@newton.ist.ucf.edu:MemCam/testbeds/context_memory_180s/manifest.jsonl \
  /data/ab575577/MemCam/outputs/external_baselines_180s/inputs/manifest.newton.jsonl
```

Use structured JSON parsing to remap only filesystem locations such as
`input_image`, `pose_path`, `gt_frames_dir`, and `overlap_dir`. Preserve scene,
start frame, row order, prompt, caption key, split identity, FPS, chunk count,
duration, frame count, and output prefix. Keep the original file and its SHA256.
Also save a path-independent identity hash so the two machines can be checked.
Do not regenerate the split: a different eligibility filter can change starts.

Important manifest fields are `scene`, `start_frame`, `duration_sec`, `fps`,
`num_frames`, `input_image`, `pose_path`, `gt_frames_dir`, `prompt`,
`output_prefix`, `split_id`, and `split_seed`. Preserve any additional fields.
Expected output names are `output_prefix + "custom.mp4"`, under separate run
directories. Do not identify the cohort by taking the first 15 files in a folder.

The separate 60-second manifest is `testbeds/context_memory/manifest.jsonl`;
its relevant rows are `3,8,13,18,23,28,33,38,43,48,53,58,63,68,73`.
It has different trajectory starts. A 60-second prefix of our 180-second videos
is not automatically the existing 60-second cohort.

## 4. Input and Frame Alignment Contract

### Camera Intrinsics: Published Renderer Settings Found

The Context-as-Memory paper's **Appendix B, Camera Trajectories**, reports
**24 mm focal length, aperture 10, and FOV 52.67 degrees** for dataset rendering.
Use this primary source rather than guessing from MemCam's retrieval settings.
[Dataset rendering settings](https://arxiv.org/html/2506.03141v2#A2)

The local MemCam code does not establish a calibrated image intrinsic matrix:

- `dataset/poses.py` reads position and rotation to construct extrinsics.
  The bundled `assets/test.json` contains position, rotation, and scale.
- `diffsynth/pipelines/wan_video_memcam.py` sets `FOV_HALF_H=45.0` and
  `FOV_HALF_V=30.0` for its overlap retriever. These are not the published
  rendering FOV and must not be reused as verified dataset intrinsics.
- `utils/prepare_worldscore_context_memory.py` writes `focal_length=500`;
  this hard-coded adapter value is not renderer evidence either.

The paper does not explicitly name the FOV axis in that sentence. Unreal's
camera API documents `field_of_view` as horizontal and exposes
`current_horizontal_fov` for cine cameras. Interpreting 52.67 degrees as
horizontal is therefore a source-supported **inference**, not a per-file
calibration already verified on CECSL. Also check aspect-ratio constraints,
cropping, and any metadata shipped with the local dataset.
[Unreal cine-camera documentation](https://dev.epicgames.com/documentation/en-us/unreal-engine/python-api/class/CineCameraComponent?application_version=5.0)

For a smoke test, derive nominal intrinsics under explicit assumptions: the
native image covers the stated horizontal FOV, square pixels, centered principal
point, and no lens distortion. For native width W0 and height H0:

```text
f = W0 / (2 * tan(radians(52.67) / 2))
K0 = [[f, 0, W0/2], [0, f, H0/2], [0, 0, 1]]
f / W0 = approximately 1.01011919
```

Read W0 and H0 from the actual input PNG, not MemCam's output resolution. Apply
the real pixel-coordinate crop/resize/padding transform A: `K_input = A @ K0`.
Account for the preprocessing library's pixel-center convention. Normalize only
as the installed DFoT loader expects; do not feed millimeters into pixel focal
length fields or force fx=fy after anisotropic resizing.

The inspected public DFoT `CameraPose.from_vectors` expects **16 values**:
four normalized intrinsics `(fx/W, fy/H, cx/W, cy/H)` for the preprocessed image,
followed by the flattened 3x4 **world-to-camera** matrix. Its ray code uses
half-pixel centers. Verify the installed version agrees, and do not pass
MemCam's 12-value c2w representation directly into this interface.
[DFoT camera representation](https://github.com/kwsong0113/diffusion-forcing-transformer/blob/main/utils/geometry_utils.py)

Record the paper URL, FOV-axis inference, native dimensions, preprocessing
transform, resulting K, and assumptions in the study configuration. This permits
a clearly labeled engineering smoke test; it does not prove that local images
are uncropped or every exported frame has identical calibrated intrinsics.
Inspect local metadata and input/output alignment before the full run. Do not
modify MemCam's retriever constants or relabel these nominal settings as measured.

### Matching Generation Inputs

For each model, implement a manifest-driven batch adapter, not a demo-video
launcher. Audit these rules before the full run:

1. Use the manifest's initial image and supported prompt conditioning. Verify
   the actual MemCam input contract in `utils/run_context_memory_batch.py`.
   If a model requires additional observed GT frames, report that difference
   before running; do not quietly give it extra scene observations.
2. Supply the same numerical trajectory wherever the model supports camera
   conditioning. Inspect `dataset/poses.py` and the receiving model's loader
   for c2w/w2c, axis conventions, translation units, reference-camera
   normalization, intrinsics, and crop/resize effects. Test a known rotation
   and translation and inspect the converted path before generation.
3. Read future poses and prompts as permitted conditioning, but never future
   GT images. After initialization, use generated history for continuation;
   do not restart every chunk from a GT image. Generated keyframe interpolation
   is distinct from interpolation conditioned on future GT endpoints.
4. Record generation seed separately from manifest split seed. MemCam's batch
   runner defaults to generation seed 42, while `seed0` in filenames identifies
   the split. Inspect saved configs to confirm the actual run seed. The same
   integer across different models does not mean identical noise realizations.
5. Preserve each method's documented inference procedure and compatible
   checkpoint. Record steps, guidance, context schedule, chunk overlap,
   resolution, dtype, and acceleration settings. Do not force a MemCam-specific
   76-frame chunk or B32 archive onto an unrelated architecture.

The current MemCam manifests use 30 FPS, with **5,397 frames for nominal 180s**
and **1,825 frames for nominal 60s**, due to chunk rounding. Read the manifest
and record actual frame count and timestamps; do not assume 5,400 or 1,800.
MemCam's usual output resolution is 640x352; verify existing run metadata.

For a compatible frame-indexed run, output frame `k` corresponds to GT dataset
frame `start_frame + k`. Verify whether output includes the initial image, chunk
overlaps, and the last partial chunk. Save an explicit index/timestamp map.

If a baseline uses another native temporal sampling rate, implement and document
a common physical-time evaluation map before scoring. Merely relabeling FPS,
padding repeated frames, or inventing intermediate frames is not alignment.
Changing the evaluation sampling requires evaluating MemCam under that same
protocol too; old scores then cannot simply be copied into the new comparison.
Apply the same principle to spatial crops. Save untouched native outputs.

## 5. Evaluation: Reuse the Existing Protocol

Primary outputs are LPIPS, FVD, and the six standard VBench dimensions:

```text
subject_consistency       background_consistency
motion_smoothness         dynamic_degree
aesthetic_quality         imaging_quality
```

Reuse the verified MemCam evaluators, detector weights, preprocessing, sampling,
and aggregation. Copy the existing metric configurations with the inputs; code
defaults alone do not establish which settings produced the headline numbers.

For the existing aligned 30-FPS protocol, the quality evaluator is
`utils/evaluate_context_memory_prefix_curves.py`: LPIPS at frame stride 30 and
224-pixel inputs; StyleGAN-V I3D FVD with 16-frame clips, four clips per video,
frame stride four, and 224-pixel inputs. Verify these against the source results.
FVD is computed across the matched cohort's clips, not averaged per-video FVD.

Once the receiving session has created the remapped manifest and validated a
model's complete aligned videos, this is the existing evaluation command pattern
from `~/MemCam` (example run name `dfot`; replace with the verified variant name):

```bash
STUDY=/data/ab575577/MemCam/outputs/external_baselines_180s
RUN=dfot
"$HOME/.conda/envs/memcam/bin/python" -u utils/evaluate_context_memory_prefix_curves.py \
  --manifest "$STUDY/inputs/manifest.cecsl.jsonl" \
  --dataset_root /data/ab575577/Context-as-Memory-Dataset/Context-as-Memory-Dataset \
  --model_output_dir "$STUDY/$RUN" --metrics_dir "$STUDY/metrics/quality" \
  --run_name "$RUN" --source_duration 180 --eval_durations 180 \
  --learned_metrics lpips,fvd --metric_device cuda --metric_batch_size 8 \
  --frame_stride 30 --learned_image_size 224 \
  --fvd_backend styleganv_i3d --fvd_clip_length 16 --fvd_clips_per_video 4 \
  --fvd_frame_stride 4 --fvd_image_size 224 --strict
```

This command is not a substitute for implementing a native-FPS or camera-input
adapter. Retain all evaluator configuration and frame-level outputs where useful.

Run standard VBench's `evaluate.py` in the existing `vbench` environment using
`--mode custom_input` and these six dimensions. Stage exactly the 15 intended
videos, without previews or partial files. Reuse identity/score validation from
`utils/run_budget_metric_grid.py:validate_bench` after checking its interface.
Each dimension must contain those same 15 identities. Keep scores in [0,1] in
the exported CSV, with percentages only for display. Handle imaging-quality
per-video scaling exactly as the existing validator does; do not divide twice.

VBench-Long is a separate evaluator and separate result family, not another name
for standard VBench. Make it an optional follow-on in the same launcher, using
the existing wrapper and matching its saved configuration. CUT3R calibration
remains unresolved in this project; do not make it a prerequisite for finishing
this external comparison. Do not mix 60-second VBench with 180-second FVD in one
apparently matched row.

## 6. Measure Resources Without Mixing Different Quantities

Record GPU model, CPU model, environment, wall time, generated frames/seconds,
peak CUDA allocated/reserved memory, and peak process RSS during generation.
Synchronize CUDA around timed GPU phases. Separate startup/checkpoint loading,
steady generation, and video writing; report the timing boundary explicitly.

Keep each method's native context representation. Report persistent RGB frames,
latent history, and conditioning tokens in their own units where available.
A packed context is not automatically a 32-frame archive. Separate the memory
used for future conditioning from decoded video buffers saved only for output.

MemCam's existing CPU query benchmark measures retrieval only: one 180-second
trajectory, eight sampled queries, one timed repeat. It is not end-to-end video
generation time and cannot be placed in a column containing FramePack diffusion
time. If a model has no discrete retrieval operation, its retrieval latency is
N/A, not zero. New cross-model speed comparisons need a shared hardware/protocol.

## 7. Implement One Resumable End-to-End Launcher

After the implementation/checkpoint audit, deliver one shell entry point with
explicit config paths and GPU selection. It must:

1. Validate the manifest, all initial images/poses/GT paths, checkpoints,
   environments, disk space, and the planned cohort before expensive work.
2. Run one short integration check per model, inspect input/output alignment,
   and estimate full-run cost from measured progress. Use a separate smoke
   directory; a short smoke output must never count as a completed 180s video.
3. Generate every missing matched video, then automatically evaluate completed
   cohorts and export tables. After the user approves the protocol/time budget,
   this must not require requesting a new command for each stage.
4. Resume completed compatible work. A filename alone is insufficient: check
   frame count, decode success, seed, input identity, checkpoint and config.
   Write outputs atomically and preserve failure logs. Do not overwrite old
   MemCam results or unrelated videos.
5. Run concurrent workers only on explicitly assigned GPUs. Keep metric jobs
   from competing for the generation GPU. Print per-video progress, elapsed
   time, failures, and the remaining workload.
6. Return a nonzero final exit code if requested work is incomplete, while
   exporting a useful coverage report and preserving successful results.

Use upstream inference code for the model implementations. Add focused tests
for manifest remapping, camera conversion, frame alignment, no future-GT access,
resume validation, exact-cohort metric collection, and failure propagation.

## 8. Deliverables and First Reply

Return actual paths to videos, configs, logs, and these exports:

- `coverage.csv`: each model/trajectory, generation status, frame count, and
  metric completeness, with missing or failed cases explicitly identified.
- `scores.csv` and `comparison.tex`: model/variant, checkpoint, camera-input
  status, initial observed-frame count, nominal/actual duration, N, LPIPS,
  FVD, six VBench scores, and clearly named resource measurements.
- `provenance.json`: code commits, checkpoint hashes/revisions, original and
  remapped manifest identities, frame/camera transforms, generation settings,
  metric configurations, hardware, environments, and source result paths.

Reuse existing MemCam and MemCam + KEEPSAKE rows only after verifying their
cohort and metric protocol. Historical headline values (FVD 734.2 vs. 476.6;
LPIPS 0.5980 vs. 0.5876) are reference checks, not hard-coded export inputs.
Leave unavailable metrics missing. Preserve results regardless of which model
wins. This is an external model comparison, not proof that a change of retention
rule alone caused differences between different pretrained generators.

In your first reply, give a short factual inventory: dataset/manifest found,
DFoT implementation/checkpoint found, FramePack implementation/checkpoint found,
which accept poses, and available GPUs. Then implement the compatible adapters
and launcher. If a checkpoint or camera-conditioned implementation is missing,
name that exact blocker and the viable alternative. Do not substitute a demo
on another dataset or claim the full study has run before it has.
