# WorldMem Handoff: Retrieval Deterioration Diagnostic

## Task

Reproduce the MemCam retrieval-deterioration diagnostic in WorldMem using actual
logged reads and the corresponding generated observations and ground truth.
Do not assume WorldMem must reproduce MemCam's trend. Preserve null or reversed
results. This is a quantitative temporal diagnostic, not a search for selected
video examples and not a causal snowballing experiment.

The figure has two panels:

1. Early-to-late changes in selected-view mismatch and selected-memory corruption.
2. Selected-memory effective mismatch versus the hindsight-best available
   effective mismatch over rollout time.

## Reference Code

- `utils/analyze_retrieval_quality_decomposition.py`: original feature extraction,
  retrieval-trace reconstruction, and query-level measurements.
- `paper/plot_retrieval_deterioration.py`: new CPU-only aggregation and figure.
- `tests/test_retrieval_deterioration.py`: synthetic aggregation, input validation,
  and end-to-end plotting tests. These do not validate the WorldMem data adapter.

The new figure does not hardcode the manuscript's estimates. It reads:

```text
~/memcam_results/context_180s/unbounded_failure_decomposition_180s/tables/query_decomposition.csv
```

The actual run contained 15 trajectories, 4,200 selected query records, and
1,050 trajectory-section pairs: 70 sections per trajectory, four sampled queries
per section on average. This is the 180-second diagnostic, not the 60-second
primary metric suite.

## Stage 1: Define Each Retrieval Event

For target frame q, identify:

- i: the historical frame actually selected by the retriever.
- H(q): the history eligible for retrieval at that instant, before generating
  the current chunk. Exclude future frames and anything the reader cannot use.
- xhat_i: the generated image corresponding to the stored observation i.
- xGT_i: ground truth at that observation's own historical trajectory index.
- xGT_q: ground truth for the current target index.

MemCam's trace reader uses selected `context_access` events, keyed by
`(section_idx, target_frame)`, with logged `selected_memory_frame`. The analysis
samples events whose `context_slot % 19 == 0`; the fallback context slot is
`target_frame % 76`. Sampling is independent of measured image quality.

MemCam reconstructs each pre-read bank from insertions and eviction events.
For chunk/section s with stride 76, it excludes the four continuation frames
`76*s - 3` through `76*s` from the bank presented to the memory retriever.
Its full eligible-history comparison is `range(0, max(0, 76*s - 3))`.
These are MemCam implementation details: do not copy them into WorldMem.
Use WorldMem's actual pre-read candidates, eligibility rules, and chunk timing.

The original analysis checks the scene, dataset start index, and duration against
trace identity. In strict mode, a selected frame missing from the reconstructed
bank or a reconstructed/logged candidate-count mismatch causes an error.

## Stage 2: Extract Features

The reference encoder uses `facebook/dinov2-base` via Hugging Face
`AutoImageProcessor` and `AutoModel`, with the checkpoint's image preprocessing.
The model runs in evaluation/inference mode. It takes `pooler_output` when
available, otherwise `last_hidden_state[:, 0]`, then L2-normalizes the float32
feature vector. Features are cached separately for generated and GT frames.

For normalized vectors a and b, the distance is:

```text
d(a, b) = 1 - clip(dot(a, b), -1, 1)
```

Use identical encoder weights and preprocessing for generated images and GT.
Feature extraction may use a GPU; cached features and scalar CSV post-processing
do not need a GPU. The current figure script itself does not import Torch,
decode videos, extract features, launch generation, or submit jobs.

The old feature cache primarily checks available feature count. For a new
WorldMem implementation, strengthen cache identity with source-video/GT identity,
frame mapping, checkpoint revision, preprocessing configuration, and hashes where
feasible. Count alone cannot establish that cached features belong to a video.

## Stage 3: Four Query-Level Measurements

Compute these on the same retrieval event:

```text
selected_view_mismatch
    = d(feature(xGT_i), feature(xGT_q))

selected_memory_corruption
    = d(feature(xhat_i), feature(xGT_i))

selected_effective_mismatch
    = d(feature(xhat_i), feature(xGT_q))

full_oracle_effective_mismatch
    = min over j in H(q) of d(feature(xhat_j), feature(xGT_q))
```

Interpretation:

- View mismatch compares clean views: was the selected historical index a good
  match for the requested target view? This is a DINO appearance proxy, not a
  direct camera-pose error.
- Corruption compares generated content with its own historical GT: did the
  stored observation depict what should have been there? It includes incorrect
  structure/content and camera-following errors, not just blur or artifacts.
- Effective mismatch measures the retrieved generated image against target GT.
- The oracle is the best generated observation in eligible history, NOT the best
  GT image, and NOT the best frame anywhere in the completed future rollout.

The oracle uses GT only for offline diagnosis. It is not a deployable retriever.
The first two distances do not add up to the third: cosine distances are not an
additive attribution of error. The vertical selected-oracle gap is diagnostic
headroom under this feature metric, not a proven achievable generation gain.

## Stage 4: CSV Contract and Validation

Required columns for the current CPU plotter:

```text
run_name,row,scene,dataset_start_frame,duration_sec,section_idx,target_frame,
selected_view_mismatch,selected_memory_corruption,selected_effective_mismatch,
full_oracle_effective_mismatch
```

Optional checked column: `candidate_count_mismatch`, which must be zero when
present. Preserve additional provenance such as selected memory IDs, eligible
candidate IDs/count, oracle ID, timestamps, seed, and source paths in the exporter.

The plotter filters one run and duration. Trajectory identity is the tuple
`(row, scene, dataset_start_frame, duration_sec)`; query identity additionally
includes section and target frame. It rejects duplicate queries, nonfinite
distances, oracle mismatch greater than selected mismatch by more than 1e-5,
wrong trajectory count, noncontiguous section coverage, fewer than eight sections,
and differing section index sets across trajectories. It does not silently
intersect partial trajectories or allow the plotted cohort to change over time.

Limit: it validates recorded CSV identities and values, not source-video hashes,
GT correctness, all query-slot coverage, or the upstream DINO computation.
Audit those in the exporter. If WorldMem includes multiple seeds, extend the
trajectory identity to include seed (or use a manifest row that uniquely identifies
each rollout); never collapse different rollouts into one trajectory.

## Stage 5: Aggregation

Let A[v,s,m] be metric m averaged over sampled retrieval queries in section s of
trajectory v. All sections are equally weighted within a trajectory, regardless
of their query count. All trajectories are equally weighted in cohort summaries.

For S available contiguous sections, let K = floor(S/4). For each trajectory:

```text
delta[v,m] = mean(A[v, last K sections, m])
             - mean(A[v, first K sections, m])
```

With 70 sections, early and late each use 17 sections. The middle sections do not
enter this contrast. They DO enter the time curves. The summary is the mean of
the 15 trajectory-level changes, not a bootstrap of 4,200 independent queries.

For the time curves, use `np.array_split` to divide ordered sections into eight
contiguous, approximately equal-count bins. First average sections within each
trajectory/bin, then average trajectories. The x coordinate is the mean sampled
target-frame index in that bin divided by FPS, not a nominal chunk boundary.
In MemCam, FPS is an explicit CLI input, default 30. WorldMem must supply its
actual frame/timestamp mapping, especially if latent and RGB rates differ.

## Stage 6: Confidence Intervals

Use 10,000 bootstrap draws with `np.random.default_rng(0)`. Each draw samples N
whole trajectories with replacement, keeping all metrics and time bins together.
Compute the mean on each draw. The 2.5th and 97.5th percentiles form a 95% interval.

Use the same trajectory resampling indices for all metrics and bins. Curve bands
are pointwise intervals, not simultaneous confidence bands. The plot does not
perform a global trend test or infer equivalence when an interval contains zero.
Do not infer significance of a red-blue contrast just by comparing their bands;
bootstrap the paired difference if that contrast is to be tested explicitly.

## Stage 7: Plot and Outputs

Left panel: view-mismatch and corruption late-minus-early changes. Faint points
show individual trajectory changes with vertical separation for visibility;
large points show cohort means; horizontal lines show bootstrap intervals.
A dashed vertical zero line separates improvement from deterioration.

Right panel: selected effective mismatch in red, eligible-history oracle mismatch
in blue, with pointwise trajectory-bootstrap bands. Lower is better for both.
All sampled sections contribute; there is no search for favorable timestamps.

Outputs: `retrieval_deterioration.png`, `.pdf`, `changes.csv`,
`trajectory_changes.csv`, `curves.csv`, `caption.txt`, and `provenance.json`.
Provenance includes the source CSV SHA256, parameters, cohort identities, query
and section counts, early/late section indices, and interpretation limits.

The four tests cover known synthetic changes and weighting, rejection of malformed
or inconsistent records, run/duration/cohort filtering, and CLI artifact export.

## WorldMem-Specific Adaptation

Before implementing, inspect WorldMem's actual memory reader and logs. Answer:
What is a stored item? What is a query? Which items were eligible? Which items
actually reached the generator? Does a read select one item, a set, or soft memory
attention? Can each item be mapped to its historical RGB frame(s) and matched GT?

If one item is selected, use the formulas above directly. If a read selects K
items, do not call an arbitrary member or the highest-attention member the
retrieval winner. Define the selected-set statistic explicitly, for example
the uniform mean of the three distances over the K actually selected items.
Aggregate the set to one query record before the existing section averaging.
The full-history minimum remains an optimistic single-item diagnostic, but the
gap then also reflects comparing a set mean to a minimum. Label it accordingly;
consider an additional best-K eligible-set mean to match the selected-set size.

If the reader uses soft attention without discrete selection, an attention-weighted
statistic is a DIFFERENT diagnostic. Specify how heads/layers/tokens are reduced,
and do not present it as an exact replication or attention as causal importance.

If a memory item is a latent, determine whether the recorded latent can be decoded
faithfully to RGB. Do not compare raw latent vectors to RGB DINO features. If using
an associated generated RGB frame as a proxy, label it as a proxy: stored latent
content can differ due to encoding, updates, or temporal compression. For blocks,
specify the constituent-frame mapping and aggregation. Audit exact GT alignment
before interpreting mismatches; no generated-to-generated substitution for GT.

Use the unbounded WorldMem rollout for this first diagnostic. No policy comparison
is needed. Additional policy curves can be added later with matched trajectories
and the same definitions. They include policy-dependent generation history and
must not be called fixed-history causal interventions.

Example after exporting a compatible WorldMem CSV (adjust cohort, duration, FPS):

```bash
python paper/plot_retrieval_deterioration.py \
  --input /path/to/worldmem/query_decomposition.csv \
  --run baseline --duration 60 --expected-videos 15 --fps 30 \
  --output /path/to/worldmem/retrieval_deterioration
```

WorldMem need not have 70 sections or four queries per section. Use its real
structure and disclose sampling. The current plotter requires common contiguous
section coverage; change this deliberately if WorldMem's evaluation design needs
a different time alignment, recording coverage rather than imputing missing data.

## MemCam Reference Result, Not a WorldMem Target

| Metric | Late minus early | 95% trajectory-bootstrap interval |
| --- | ---: | --- |
| View mismatch | -0.0335 | [-0.0511, -0.0171] |
| Stored-content corruption | +0.0873 | [+0.0221, +0.1555] |
| Selected effective mismatch | +0.0732 | [+0.0157, +0.1312] |
| Best available mismatch | +0.0008 | [-0.0496, +0.0546] |

This supports increasing selected-memory corruption despite improved selected
clean-view alignment, with no detected mean early-to-late change in best-available
mismatch. It does not prove the latter is exactly constant, that candidate count
causes deterioration, that bad reads cause bad future writes, or that a particular
retention policy prevents a causal snowball. Report the actual WorldMem outcome.
