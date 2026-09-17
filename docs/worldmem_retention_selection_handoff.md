# WorldMem Handoff: Retention and Selection Gaps

## Request to the WorldMem Agent

Build a reproducible retention-selection analysis on the matched 15-video,
60-second WorldMem suite. Audit inputs first. Use real historical candidate
banks and real selected memory IDs, evaluate all policies using one common
source of historical pixels, and export data-driven figures and tables.

This is NOT the earlier temporal retrieval-deterioration diagnostic, NOT a
latency benchmark, and NOT another search for a snowballing example. It asks:

1. How much target-relevant evidence became unavailable through eviction?
2. How much target-relevant evidence remained available but was not selected?

Do not assume Ours wins either component or their sum. Retain unfavorable and
null results. Do not launch new generation before determining which existing
traces and feature caches are usable.

## MemCam Reference and Current Scope

Reference files in the MemCam repository:

- `utils/analyze_common_source_retention_selection_budget.py`: shared-source scoring.
- `utils/analyze_retrieval_quality_decomposition.py`: feature encoder, bank reconstruction and single-item gaps.
- `paper/audit_gap_inputs.py`: CPU audit of matched traces and feature caches.
- `docs/retention_selection_analysis_plan.md`: interpretation and figure plan.

Reuse the concepts, not MemCam's chunk stride, continuation-frame exclusions,
trace field names, or single-item selection assumption. MemCam retrieves one
historical frame per target-frame query; WorldMem reportedly selects eight.

MemCam's old figure used a 13-trajectory, 180-second subset. RI-B128 has only six
180-second video/trace files on Newton, but fifteen at 60 seconds. No saved
RI-B128 gap summary was found by the recent search. We are now auditing the
15-trajectory, 60-second grid so the primary diagnostic can cover every policy
and budget. Do not mix the old 180-second numbers with new 60-second results.

The intended WorldMem grid is Unbounded plus FIFO, RI, Ours, K-center and MCE at
B=16,32,64,128: 21 configurations if all are actually implemented and available.
Use the WorldMem implementations and actual run names. List unsupported or
missing configurations explicitly; do not pretend a similarly named MemCam
policy has already been transferred.

## Stage 1: Inspect the Reader and Establish Identity

Before writing the adapter, inspect the production WorldMem reader and logs.
Write down code references and answers to these questions:

- What is one stored item: RGB frame, latent frame, temporal block, or updated state?
- At what instant is a read made, and what target view/frame does it serve?
- Are eight distinct memory IDs selected? Are repeated IDs or padding possible?
- Are there forced context slots, temporal exclusions, or constraints on sets?
- Which selected items actually enter generation, versus only being scored?
- Does the logged bank describe state before retrieval or after the current write?
- Can every item be mapped to an exact historical index in common-source RGB?
- Are initial frames or endpoints protected? Does the same rule apply to all policies?
- Does B count all persistent retrievable items, including protected initial context?

The user-confirmed unbounded protocol has 600 context frames, then generates
600 more at 10 FPS. Generated time is 0..60 seconds, not the age of the complete
history. The existing workload analysis reports eligible counts 600+k at
generated index k=0..599 and eight selected memories per query. Verify these
identities against the reader; do not insert these counts as synthetic bank logs.

The bounded policies need separate checks. Do not assume the initial 600 frames
are all retained outside B, all evictable, or protected in the same way. If an
auxiliary retrievable store exists, disclose it and include its effects. A bank
of 32 plus an uncounted initial store is not a total 32-item archive.

Record a manifest identifying each rollout by scene, trajectory/start index,
seed, duration, FPS, prompt/config, checkpoint and source paths. Fifteen videos
means the same fifteen identities across policies, not any fifteen filenames.
The previously reported unbounded run was
`worldmem_memquality_unbounded_60s_n15_seed101`; confirm it is the matching source.

## Stage 2: Construct the Per-Query Sets

For policy p and target query q, define:

- H(q): all historical observation IDs eligible under the reader's query-time
  rules if none had been evicted. Include eligible initial context; exclude
  current/future outputs and anything the reader is prohibited from reading.
- M(p,q): the actually retained eligible candidate bank immediately before q.
- R(p,q): the K IDs actually selected from that bank and supplied to generation.

Require M(p,q) subset H(q), and R(p,q) subset M(p,q). Reconstruct the bank from
initialization, insertions, evictions and eligibility filters. Compare its count
with the logged candidate count. A final archive snapshot is not sufficient.

H(q) must be common across policies when historical indices and query timing
are shared. If policy-dependent item representations prevent this, stop and
describe the mismatch rather than manufacturing common frame IDs.

The common source fixes evaluation pixels; banks and selected IDs still come
from each policy's own generation trace. We are NOT claiming that every policy
made admission/eviction decisions on one frozen generated stream. Such an
offline policy replay would be a separately labeled experiment.

## Stage 3: Fix Pixels and Extract Features

For historical ID i, use the associated image from ONE common unbounded rollout,
denoted x_src[i]. Use GT at the target's exact dataset index, x_GT[q]. Do not
use each policy's own RGB when comparing the banks in this experiment.

Initial context requires explicit handling. If those 600 frames were observed
inputs rather than generated frames, use the actual input images and label
their origin. Do not replace generated memories with GT. Do not score initial
frames using an arbitrary PSNR ceiling or claim they were generated.

Map generated-relative indices, full-history indices and dataset indices
separately. In the reported setup a generated index k may map to history ID
600+k; verify this rather than confusing k with its GT dataset index.

For consistency with MemCam:

- Encoder: `facebook/dinov2-base`, Hugging Face image processor and model.
- Run evaluation/inference mode with identical preprocessing for source and GT.
- Take `pooler_output` when available, otherwise `last_hidden_state[:, 0]`.
- Convert to float32 and L2-normalize each feature.
- Define d(i,q) = 1 - clip(dot(phi(x_src[i]), phi(x_GT[q])), -1, 1).

This is target-appearance mismatch. It includes wrong content/view, not merely
blur or aesthetic quality. It is not a true label of downstream conditioning
utility. The same-index generated-versus-GT corruption diagnostic is different.

For latent memories, associated RGB may be a proxy rather than the stored
latent's exact content. State that explicitly. If items are compressed blocks,
define their frame mapping and feature aggregation before computing gaps.
If an updated/merged item cannot be mapped to common historical content, the
frame-index protocol cannot be claimed as an exact replication.

Cache source and GT features separately with source identity/hash, index map,
encoder revision, processor configuration and feature shape/dtype. Check finite,
normalized features and required frame coverage. A file with the right length
is not automatically the right cache. Feature extraction may need a GPU; the
cached-distance and gap calculations are CPU-only.

## Stage 4: Correct Eight-Item Decomposition

For K distinct unconstrained selections, define the set score

```text
J(S,q) = mean(d(i,q) for i in S), where |S| = K.

a(q)   = mean of K smallest distances among H(q)
b(p,q) = mean of K smallest distances among M(p,q)
c(p,q) = mean distance of the actual selected IDs R(p,q)

retention_gap(p,q) = b(p,q) - a(q)
selection_gap(p,q) = c(p,q) - b(p,q)
total_gap(p,q)     = c(p,q) - a(q)
```

For WorldMem, use K=8 after verifying the reader. MemCam is the K=1 special
case. Do NOT compare the mean of eight selected memories with a single best
historical item. Do NOT turn the highest-attention member into the winner.

Require |H| >= K, |M| >= K, distinct selected IDs, and a <= b <= c up to floating
point tolerance. The sum of the two gaps must equal c-a. Fail on material
violations rather than silently applying max(0, gap). Numerical discrepancies
below a declared tolerance such as 1e-5 can be handled explicitly and counted.

If there are forced slots or hard constraints on valid selected sets, replace
the naive K-smallest oracle with the minimum J over feasible K-item sets. Use
the same query-specific feasibility rules in both histories, with the retained
bank merely restricting available IDs. This ensures nested feasible families.
For duplicates or variable K, define a compatible multiset/variable-size protocol
first. Do not silently drop difficult queries or relabel a relaxed oracle as
the reader's feasible best. A soft-attention-only reader requires a different
diagnostic and must not be passed through this discrete-selection formula.

Even a valid best-K mean oracle measures independent appearance distances; it
does not capture complementary views or nonlinear interactions in the generator.

## Stage 5: Interpretation Checks

Example: a=0.20, b=0.25, c=0.40 gives retention=0.05, selection=0.15, total=0.20.

- Retention gap: how much target-appearance quality is lost from the best feasible
  evidence when the archive is restricted to the retained bank.
- Selection gap: how far the actual selection falls short of the best feasible
  selection still available inside that bank.
- Total gap: actual selection's shortfall from the complete-history oracle.

Unbounded has zero retention gap when M=H. Selecting the retained oracle gives
zero selection gap. A bank containing exactly K items can have zero selection
gap while every one of those items is poor. Always show total gap and preserve
absolute a,b,c values. Selection gap can rise solely because the bank oracle
improves, even if actual selected quality is unchanged.

Lower gaps do not prove improved downstream generation, lower corruption, or
causal snowball prevention. Common-source results do not identify the full
causal contribution of protected initial memories. Audit protection rules and
export selected initial-context counts so their role can be examined without
concealing those selections or silently removing them from aggregates.

## Stage 6: Matched Cohort, Queries and Aggregation

Audit all requested runs before analysis. Report each policy/budget's matched
videos, valid traces, query coverage, and needed source/GT caches. A completed
LPIPS/FVD/VBench result does not establish bank reconstruction availability.

Freeze the fifteen trajectories and common target-query IDs. Prefer all actual
generated-frame reads at 10 FPS if practical. If subsampling, choose fixed query
indices independently of output quality and use them for every policy/budget.
State the exact sampling rule. Missing required queries are incomplete inputs,
not permission to silently intersect away inconvenient observations.

Aggregate the K selected memories to one scalar per query first. Then average
queries within each trajectory and weight the fifteen trajectories equally.
This matches the retention-selection analysis's trajectory weighting. The other
retrieval-deterioration diagnostic uses section-first averaging; do not conflate
the two procedures. Record query counts per trajectory.

For uncertainty, bootstrap whole trajectory IDs with replacement, 5,000 draws,
seed 17, and take the 2.5/97.5 percentiles. Use the same resampling indices across
policies, budgets and components. Compute total-gap intervals from trajectory
totals, not by adding endpoints of component intervals. Use paired trajectory
differences to assess policy contrasts; overlapping marginal CIs are not a test.
Do not treat eight memories or hundreds of queries as independent videos.

## Stage 7: Files to Export

`coverage.csv`: one row per policy/budget/trajectory, with missing paths, invalid
reads, reconstruction mismatches and cache status. Do not hide failures in a
figure-generation log.

`query_gaps.csv`: one row per policy/trajectory/query. Recommended columns:

```text
system,run_name,policy,budget,trajectory_id,scene,seed,dataset_start_frame,
duration_sec,query_id,target_generated_frame,target_dataset_frame,
generated_time_sec,section_idx,k,eligible_count,retained_count,
full_oracle_distance,bank_oracle_distance,selected_distance,
retention_gap,selection_gap,total_gap,selected_initial_context_count,
bank_snapshot_id,source_id
```

Use `selection_gap` consistently in new files; MemCam's older CSV calls this
`retrieval_gap`. Document the alias when combining outputs.

Save initial IDs, writes, removals, eligible/retained snapshots, actual selected
IDs and both oracle ID sets in JSON/JSONL sidecars. Bank IDs can reference shared
snapshots to avoid duplicating large arrays in every CSV row. Preserve ordering
and tie-breaking information; arbitrary tied oracle choices must not change
the scalar result.

`trajectory_gaps.csv`: query count and means of a,b,c and all gaps per trajectory.

`summary.csv`: system, policy, budget, N, horizon, K, component/total means and
bootstrap intervals. Include exact source/cohort identifiers, not only run names.

`provenance.json`: manifest identities; checkpoint/config; protected-item rules;
query sampling; input, cache and adapter hashes; feature configuration; item/GT
mapping; selection/oracle constraints; aggregation; bootstrap settings; exclusions;
and whether RGB is a proxy for latent memory. Separate observed and assumed facts.

## Stage 8: Figures and Tests

Main-paper figure: matched B32 horizontal stacked bars in the fixed order
Unbounded, FIFO, RI, K-center, MCE, Ours, including only completed audited runs.
First segment: "Lost through eviction". Second: "Retained but not selected".
The endpoint is total gap, with a trajectory-bootstrap CI. Axis: "DINO distance
gap (lower is better)". Unbounded's zero retention segment is not a missing value.

Produce a WorldMem panel with K=8 stated. It will accompany a separately labeled
MemCam K=1 panel. Do not compare their absolute gap magnitudes as a cross-model
leaderboard: initial histories, source content and selection sizes differ.
Export PDF and PNG directly from result CSVs, without hard-coded favorable values.

Appendix: full B16/32/64/128 retention-versus-selection sweep, a numerical table
including total gap, and explicit missing-data marks. Never invent or interpolate
an absent RI-B128 point. Do not omit a baseline because it beats Ours.

Tests must cover:

1. Known synthetic a,b,c values and exact telescoping identity.
2. K=1 equivalence to MemCam and K=8 best-K correctness.
3. Zero retention for M=H, zero selection for oracle selection, tied distances.
4. An exactly-K poor bank with zero selection but positive retention/total gap.
5. Rejection of duplicate/missing selected IDs, future memories, M not subset H,
   insufficient candidates, inconsistent K and invalid constrained selections.
6. Initial-context versus generated-index mapping at the 600-frame boundary.
7. Cohort/seed mismatches, incomplete query coverage and stale feature caches.
8. Equal trajectory weighting despite unequal query counts, paired resampling,
   and total CI computed from paired trajectory totals.
9. End-to-end export from synthetic logs through the real WorldMem adapter,
   followed by a small real-trace smoke test before processing all policies.

## Immediate Deliverable

First return the coverage audit and the verified reader contract, including
whether K=8, distinctness, initial-context protection and budget accounting hold.
Then run the CPU analysis if compatible caches and complete traces already exist.
If only source/GT features are missing, propose a feature-extraction job, not new
generation. If traces are missing, identify the exact runs and trajectories and
whether faithful reconstruction is possible. Obtain approval before expensive
regeneration. Supply actual WorldMem results, not numbers copied from MemCam.
