# Complete Pose--Appearance Ablation

One batch submission performs the three-setting component ablation on the matched
15-video, 180-second cohort. It is not the previous five-weight sensitivity sweep.

| Setting | Pose | Appearance | Generation |
| --- | ---: | ---: | --- |
| Appearance only | 0 | 1 | 15 new videos |
| KEEPSAKE control | 0.65 | 0.35 | Reuse the existing 15 videos |
| Pose only | 1 | 0 | 15 new videos |

All runs use B32, 50 denoising steps, the same prompts/poses and generation seed 42.
Keep the graph threshold, utility coefficients, protected-frame rules, descriptor
extraction, and retriever unchanged. This is a signal ablation, not an endpoint
runtime optimization. The original batch runner did not record generation seed in
legacy control logs: default 42 is recorded as an assumption, not certified history.
Missing legacy affinity metadata refers to the original fixed 0.65/0.35 code.

## One Submission

```bash
cd "$HOME/MemCam"
sbatch slurm/newton_keepsake_pose_appearance_complete.sbatch
```

One job, no array, no follow-up submission script. It requests two H100 GPUs for
up to 72 hours; each endpoint uses one GPU. The observed control runtime was about
four hours per 180-second video, so endpoint generation alone is roughly 120
GPU-hours / 60 elapsed hours when both GPUs run concurrently. Evaluation adds time;
72 hours is an allocation ceiling, not a completion-time guarantee.

Default output: `~/memcam_results/keepsake_pose_appearance_180s/`.
The original control files remain untouched. All generated files live under this
new experiment directory. Use `KEEPSAKE_ABLATION_OUTPUT` to change the output root.

## Automatic Workflow

1. Check the matched manifest, control video frame counts/FPS, and memory traces.
   Stage only the 15 matching control videos, excluding extras.
2. Test GPU allocation/kernel execution in both existing Python environments,
   without Conda activation hooks or a short startup timeout.
3. Evaluate the existing control with LPIPS, cohort FVD and all six standard VBench
   dimensions. Do this before spending compute on either new endpoint.
4. Generate the two endpoints concurrently, 15 videos each. Validate every completed
   video and its trace, recording content hashes and the actual affinity weight.
5. Evaluate each complete endpoint with the same LPIPS/FVD/VBench configuration.
6. Export `ablation.csv` (all six dimensions plus aggregate) and `ablation.tex`
   (pose/appearance weights, FVD, LPIPS, weighted-six VBench aggregate).

FVD uses the 15-video group with four clips/video, not averaged per-video FVD.
VBench is standard VBench, not VBench-Long. VBench dimensions and the six-dimension
aggregate in the CSV are percentages. Aggregation uses fixed official normalization
bounds and dynamic-degree weight 0.5, divided by the available weight sum 5.5.
The normalized six-dimension aggregate is not the 16-dimension total score.

## Progress and Resume

Top-level logs: `keepsake_ablation_JOBID.out` and `.err` in the submission directory.
Endpoint logs: `appearance_only.log`, `pose_only.log` in the output root.
Per-video logs: `generation/SETTING/row_NNN.log`. Metric logs:
`metrics/SETTING/quality.log` and `vbench.log`.

Each completed generation and metric stage has a receipt with source/output hashes.
Rerunning the same submission command with unchanged code and output root resumes
completed stages; it does not regenerate a control or completed endpoint video.
An interrupted video may need to restart; resumability is per video, not per chunk.
Changing frozen code/config is rejected rather than mixing experimental versions.

The driver exits nonzero on failed generation, incomplete metric coverage, a changed
source, or an evaluator failure. Failed metric stages use fresh attempt directories
on retry. Final tables are produced only after all three settings finish. Long
evaluations and generation print one-minute heartbeats with the active log path.

### GPU Preflight Failure in Job 829288

Job 829288 failed on `evc45` before control evaluation or endpoint generation.
The first selected GPU passed both environment checks. The second failed
`torch.cuda.is_available()` in `memcam`. This was not a startup timeout, and
the logs do not distinguish a device-mapping problem from a node/driver problem.
`new_videos: 30` in `status.json` is the planned workload, not a completion count.

The batch launcher now logs the Slurm allocation and CUDA mask, prints the GPU
inventory, and sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` to align CUDA and NVML ordering
as described in [Slurm's GRES documentation](https://slurm.schedmd.com/gres.html).
It does not replace the scheduler's `CUDA_VISIBLE_DEVICES` mask or remove the
kernel checks. This change is a launch precaution and diagnostic improvement,
not confirmation of the failed node's root cause.

After syncing the updated launcher to Newton, retry the same experiment while
temporarily excluding the node involved in this failure:

```bash
cd "$HOME/MemCam"
sbatch --exclude=evc33,evc40,evc43,evc44,evc45 \
  slurm/newton_keepsake_pose_appearance_complete.sbatch
```

This is still one job for both endpoints and their evaluations. The Python
experiment code and frozen configuration are unchanged, so the existing output
directory can be reused without deleting its validation records. The exclusion
does not establish that `evc45` is permanently faulty. If availability fails
again, inspect the newly logged allocation/mask and report the job/node to the
cluster administrators before another blind retry.

Local tests use synthetic files and mocked GPU commands to check the entire
orchestration, exact cohorts, metric configuration, control reuse, checkpoint reuse,
GPU separation, failure propagation, and table exports. They do not establish that
Newton's model/checkpoint installations work; the job tests them before new generation.
