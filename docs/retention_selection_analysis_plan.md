# Retention and Selection: Definition and Evaluation Plan

## What the Current Code Measures

Source: `utils/analyze_common_source_retention_selection_budget.py`, using
`compute_query_decomposition` in `utils/analyze_retrieval_quality_decomposition.py`.

For each query q, use the actual eligible history H, the policy's eligible bank
M (a subset of H), and its logged selected index r (in M). In MemCam the history
excludes the four continuation frames and all current/future generated frames.
Do not search the final completed video when defining a past query's oracle.

Every historical index is scored using generated pixels from one common source
rollout, normally unbounded. Target pixels are exact-index GT. The feature
distance is one minus the dot product of normalized DINOv2-Base features.

Three distances define the diagnostic:

- a: minimum distance to target GT over complete eligible history H.
- b: minimum distance to target GT over retained eligible bank M.
- c: distance to target GT of the actual selected historical index r.

Retention gap = b - a. Selection gap = c - b. Total gap = c - a.
Example: a=0.20, b=0.25, c=0.40 gives retention=0.05, selection=0.15,
total=0.20. A retained item tied with the history oracle gives zero retention
gap; selecting the retained oracle gives zero selection gap.

All distances use the same target, source pixels and candidate eligibility.
These are exact telescoping differences, unlike the non-additive view-mismatch
and memory-corruption distances in the separate deterioration diagnostic.
They are target-appearance proxies, not true conditioning utility or causal
attributions of final-video errors. A poor singleton bank has zero selection
gap. As a bank's oracle improves, selection gap can increase even if the actual
selected image stays unchanged. Always report total gap alongside its components.

Current aggregation: query means within a trajectory, then equal trajectory
weight. The analysis script bootstraps trajectory means with 5,000 resamples,
seed 17. This differs from the section-first aggregation used by the separate
retrieval-deterioration plotter. Do not describe the protocols as identical.

Policy banks/selected indices come from their own closed-loop traces. Fixing the
pixels used to evaluate those indices does not mean all policies made their
retention decisions on a common generated stream. A same-stream policy replay
would be a different control and must not be substituted without labeling it.

## Why RI-B128 Is Missing

The current manuscript says required traces were missing from the common matched
set. Local figure code, `paper/make_figures.py::plot_retention_selection_tradeoff`,
contains manually entered RI points for B16/32/64 only. That omission is not
evidence that B128 cannot be computed, or that newly completed videos fixed it.
The default run list of the CPU analysis already includes RI-B128 and K-center
at B16/32/64/128. No local exclusion report establishes which exact remote files
were missing; audit those before making a more specific claim.

This computation needs actual access/eviction traces and common-source/GT DINO
caches, not policy-specific LPIPS, FVD or VBench outputs. Completed videos alone
do not supply the banks that existed at each earlier read. Missing policy traces
cannot generally be recreated exactly from rendered MP4s without the original
policy inputs and decisions.

## Complete the Evidence Before Redrawing

1. Audit the 60-second primary suite and the existing 180-second diagnostic
   separately. For every baseline and B16/32/64/128 configuration, count matched
   access traces, complete query coverage, and source/GT caches. Keep seed and
   trajectory start index in the identity. Explain the existing 13/15 cohort.
2. Include unbounded, FIFO, RI, GeoCov, K-center and MCE. Hold protected-frame
   rules constant where possible; otherwise report differences and add a matched
   control. Do not infer symmetry from the fact that unbounded retains everything.
3. Fix one cohort and one sampled query list before the comparison. Do not
   silently take smaller query intersections for missing runs. The existing
   `sampled_shared_queries` intersects available reads, even in strict mode;
   explicitly validate equal query coverage before using its outputs.
4. Validate normalized finite features, source identity, exact GT alignment,
   selected membership, M subset H, and logged candidate counts. The old cache
   loader checks shape/count, not video identity. Old gap code clips negative
   differences at zero: add explicit subset/order checks rather than treating
   clipping as a substitute for validity.
5. Recompute shared-source scores from caches. Once traces and caches are
   available, this is CPU analysis, not another video generation job. MCE can be
   supplied through `--runs`, but its family name/order/plot styling needs support.
6. Export query-level results, trajectory means, paired bootstrap intervals,
   coverage and provenance. Generate figures from those files, not transcribed
   constants in `paper/make_figures.py`. Never impute missing RI-B128 points.

## Main Figure Versus Appendix

Main paper: a matched-B32 mechanism comparison, not a dense all-budget scatter.
Use one horizontal stacked bar per policy. The first segment is retention gap;
the second is selection gap. The endpoint is total gap. Label segments in plain
language: "Lost through eviction" and "Retained but not selected". State DINO
distance units and "lower is better". Unbounded has a zero retention segment,
not a missing result. Put a trajectory-bootstrap interval on the total endpoint;
bootstrap the sum per trajectory, never add independently computed CI endpoints.
Use a fixed policy order across panels; show null and unfavorable results too.

Start with one MemCam panel. Add a WorldMem panel only after its adapter is
validated. Do not substitute the current zoomed RI-versus-GeoCov subplot for a
second system. A small schematic or caption can show a -> b -> c and the two
differences; the architecture figure answers a different question.

Appendix: the full B16/32/64/128 budget sweep, with retention-vs-selection points,
explicit budget labels, total-gap values and missing-data markers. Use identical
cohorts across compared points. Retain absolute selected distance c and oracle
distance a in supplementary tables, so small gaps cannot hide a poor source
history. No need to force every policy/budget into the introductory figure.

The B32 plot is mechanism evidence. The budget sweep is sensitivity analysis;
policy replacements are controlled baselines. Neither is by itself a component
ablation or proof that memory pruning causally improves generated content.

## WorldMem Adapter

WorldMem selects eight memories, so do not choose an arbitrary member as the
retrieval winner or compare their mean to a single-item minimum. First verify
whether the eight selected IDs are distinct and which pre-read items were
eligible, including the initial 600-context-frame history.

For exactly K distinct selected items (K=8), a matched-size diagnostic is:

- a_K: mean of the K smallest target distances in complete eligible history.
- b_K: mean of the K smallest distances in the eligible retained bank.
- c_K: mean target distance of the K items actually selected.

Then retention = b_K - a_K, selection = c_K - b_K, and total = c_K - a_K.
MemCam is the K=1 special case. Require at least K eligible items, set inclusion,
and a_K <= b_K <= c_K. If duplicates, forced slots, or other set constraints
exist, use the same feasible selection rules for both oracles rather than an
invalid top-K set. Do not silently drop queries with insufficient bank size.

Use the same common-source generated pixels, exact-index GT and encoder for all
WorldMem policies within its panel. State whether decoded latents or associated
RGB frames represent a stored item; raw latents cannot be compared to RGB DINO
features. Attention-weighted selection is a different diagnostic. The mean
appearance objective does not model complementary views or the generator's
nonlinear use of the set, even when the decomposition is algebraically exact.

Keep claims within each system: different initial histories, sources, durations
and K values prevent interpreting absolute gap magnitudes as a cross-model
leaderboard. Preserve WorldMem null or reversed results rather than assuming
it will reproduce the MemCam ordering.
