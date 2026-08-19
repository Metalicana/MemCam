# Why Does Unbounded Memory Degrade With Duration? A Mechanism Hunt

Status: **open question.** Five specific hypotheses tested against real 60s
baseline traces on Newton. All five ruled out, or shown not to transfer.
This is a negative-results record, kept because ruling things out narrows
the search and because the next person (or the next session) shouldn't
re-run these same five checks from scratch.

## What's solid and not in question

The retention/retrieval decomposition (`analyze_retrieval_quality_decomposition.py`,
180s data, `unbounded_failure_decomposition_180s`) is a measured fact, not a
hypothesis:

| run | retention_gap | retrieval_gap |
|---|---|---|
| unbounded (baseline) | ~0 (by construction) | 0.2267 |
| FIFO | 0.2039 | — |
| RI | 0.0474 | 0.1582 |
| SLAM | 0.0639 | 0.1338 |

Unbounded's failure is ~100% retrieval_gap: the correct frame is always
still in memory, the frozen geometric selector just picks the wrong one
increasingly often. RI and SLAM both win by trading a little retention_gap
for a lot less retrieval_gap. **This part doesn't depend on knowing why**
retrieval_gap climbs — it's already real, already measured, already the
spine of the paper regardless of what follows below.

The open question this document is about: *why* does retrieval_gap grow as
the candidate pool grows (H1)? Five candidate mechanisms were tested. None
of them explain it.

## The five hypotheses, in the order tested

### 1. Occlusion poisoning — real, but small

**Idea:** the retriever picks by geometric FOV overlap alone, never checks
whether the region is actually visible. If a target is occluded, every
geometrically-overlapping candidate could share the same wrong, occluded
appearance.

**Tool:** `utils/diagnose_fov_occlusion_poisoning.py`. Real camera poses +
real GT-frame DINO appearance, no generated video needed. Run: baseline,
60s, 15 rows, 1380 queries.

**Result:**
- Mean `poisoning_gap` (appearance cost of the geometric argmax vs. the true
  best-appearance candidate): **0.0455** — small relative to a 0-1ish
  DINO-distance scale.
- 75.4% of queries have *some* conflict, but the size of it is what matters,
  and it's small.
- `appearance_oracle_iou_percentile` = 0.0758 — the true best-appearance
  match usually ranks *near the top* of the geometric ranking too, not
  buried at the bottom. That's evidence against severe, systematic occlusion
  poisoning as the dominant pattern.
- Pool-size correlation was initially confounded (script caps candidates at
  200/query for compute reasons; 91.3% of baseline's queries exceed that,
  so the raw pool-size number stopped reflecting the true pool past 200).
  After splitting capped-vs-uncapped: mean gap 0.0364 (small real pools) vs.
  0.0464 (large real pools) — real, ~28% relative increase, but coarse
  (binary split, not a continuous trend) and modest in size.

**Verdict:** real mechanism, modest effect, weak support for growing with
pool size. Not the dominant driver of H1 on its own.

### 2. Within-section context diversity collapse — real, wrong direction

**Idea:** the retriever computes a fresh independent argmax for each of the
~76 targets in a predicted section, with no exclusion rule. One
geometrically-central memory frame could win most or all of them, collapsing
the model's context to a handful of repeats.

**Tool:** `utils/diagnose_context_diversity_collapse.py`. Pure trace
bookkeeping — no geometry, no DINO, no GPU. Run: baseline, 60s, 15 rows, 345
sections.

**Result:**
- Mean `max_frame_share` = 0.172 (perfectly uniform would be 1/76 = 0.013;
  total collapse would be 1.0). Real concentration, not extreme.
- Mean `effective_num_frames` = 25.65 out of 76 possible — a section
  typically draws from ~26 distinct memory frames.
- 28/345 sections (8.1%) show one frame winning over half the section's
  targets — a real minority, not the norm.
- Raw Spearman(pool size, `max_frame_share`) = **−0.314**. Corrected for the
  mechanical pigeonhole floor (small pools are *forced* toward higher share
  just by having fewer distinct options) — still **−0.253**.

**Verdict:** real and moderate, but points backwards. Bigger pools show
*less* concentration, not more, even after controlling for the small-pool
floor effect. Cannot explain why unbounded gets worse with duration; if
anything this effect would predict the opposite.

### 3. Memory content drift — not happening

**Idea:** autoregressively generated frames drift from their own ground
truth the longer generation runs, so stored memory content itself degrades
over time, independent of which frame gets selected.

**Tool:** no new script — `memory_corruption` (generated frame vs. its own
GT) is already a column in the existing 180s `query_decomposition.csv`.
Zero new compute.

**Result:**
- Mean corruption, first third of the video: **0.5817**
- Mean corruption, last third of the video: **0.5641** (slightly *lower*)
- Spearman(section_idx, corruption) = 0.176 (weak, and doesn't match the
  clean first-vs-last comparison)

**Verdict:** the gap is already large from very early in the video (0.58 is
the single largest number seen in this whole investigation) and does not
grow — if anything it shrinks slightly by the end. Not a duration-driven
degradation. Ruled out.

### 4. Zero-overlap fallback laziness (WorldMem cross-system check) — real elsewhere, doesn't apply to MemCam

**Idea (from WorldMem's own analysis of their system):** ~3/4 of all
retrievals across every policy find zero genuine geometric overlap
(universal, not what separates policies). What differs is the fallback:
unbounded/FIFO always have a fresh, near-duplicate frame sitting around to
hand back (median fallback age 4-8 frames in WorldMem's data). SLAM's
redundancy-driven eviction doesn't protect recent frames, so its fallback is
forced to reach back much further (median ~81 frames in WorldMem's data) —
forced to be genuinely different, which turns out to matter more than
expected.

**Tool:** `utils/diagnose_zero_overlap_fallback.py`. Pure trace bookkeeping
(`selected_overlap` and `memory_age` are already logged per
`context_access` event in `wan_video_memcam.py`). Ran against baseline,
fifo_b32, ri_b32_dino_rgb, slam_b32_covisibility, slamri_b32_beta0p5, 60s.

**Result:**
- At `--overlap_threshold 0.01` *and* `0.1`: baseline and fifo_b32 show
  **100% hit rate** — zero misses at all, no fallback-age bucket to even
  measure.
- Overlap-value distribution for baseline: p1 = 0.334, median = 0.9335,
  exact-zero = 0.0%, below 0.3 = 0.68%, below 0.5 = 4.0%.

**Verdict:** WorldMem's mechanism is real *on WorldMem's system* but the
precondition (frequent complete misses) doesn't hold on MemCam's dataset —
median overlap here is 93%, essentially the opposite regime from WorldMem's
~75%-miss-rate finding. Not a failure of the idea, a genuine
cross-system difference (different dataset / camera trajectory / FOV
parameters). Worth remembering for any claim about this being a universal
property of FOV-based retrieval — it isn't, at least not in this form.

### 5. IoU estimation noise — real, wrong direction, and the cost is negligible anyway

**Idea:** FOV overlap is a Monte Carlo estimate (thousands of random sample
points), not exact. With most candidates already clustered near very high
overlap (see #4's distribution), the argmax "winner" among near-tied
candidates could be decided by measurement noise rather than genuine
geometric superiority — and this should get worse as the pool grows, since
more candidates means more chances for a noisy estimate to win by luck.

**Tool:** `utils/diagnose_iou_estimation_noise.py`. Recomputes the same
query's IoU at the real retriever's sample count (5000) and at 10x more
(50000), checks whether the winner changes. Pure CPU (the FOV calculation
is hardcoded to CPU regardless of device). Baseline, 60s, 15 rows, 1380
queries.

**Result:**
- Flip rate: **39.06%** of queries — measuring more carefully does change
  the winner often. This part of the theory was right.
- Mean top-2 gap at low precision: 0.0182 (confirms these are genuinely
  near-tied situations).
- Raw Spearman(pool size, flip happened) = −0.116 (wrong direction already).
- Corrected for the same 200-candidate cap as #1: **uncapped (small real
  pools, n=120) flip rate = 57.5%; capped (large real pools, n=1260) flip
  rate = 37.3%.** Confirmed, not an artifact — small pools are *more*
  unstable, not less.
- Appearance cost when the winner does flip: **0.0029** — roughly 15x
  smaller than #1's poisoning_gap. The "wrong" and "right" picks are nearly
  identical in actual content.

**Verdict:** real and common, but (a) gets *less* frequent in bigger pools,
the wrong direction to explain H1, and (b) even when it happens, costs
almost nothing in practice. Ruled out on both counts, independently.

## Open questions / what's still unresolved

- **The actual mechanism behind H1 (retrieval_gap climbing with pool size)
  is still unknown.** Five specific, reasonable hypotheses tested, five
  ruled out or shown not to transfer. That's real narrowing, not nothing —
  but it isn't an answer.
- **Pool size vs. elapsed time was never causally separated.** The H1
  tooling (`analyze_pool_growth_scaling.py`) documents this limitation in
  its own docstring: pool size and generation depth co-vary in an unbounded
  rollout, so nothing here proves it's *size* specifically rather than
  *how far into a long autoregressive rollout the generator is*, independent
  of what gets retrieved at all.
- **Whether this is even a retrieval problem at all is not fully closed.**
  Everything tested so far assumes the failure lives in *which* frame gets
  selected. It's possible some of the residual degradation is really about
  the generator's own conditioning getting harder deep into a long rollout,
  unrelated to retrieval quality — `memory_corruption` (mechanism #3) was
  the one check aimed at this and came back flat, but that doesn't rule out
  every version of the idea.
- A useful side finding, independent of any single hypothesis: MemCam's
  baseline retrieval essentially never truly misses (median overlap 93%).
  The real day-to-day question isn't "does it find something," it's "which
  of many already-good candidates wins" — every mechanism above is really a
  different theory about how that near-tied crowd gets resolved, and none
  of them turned out to explain the duration-dependent degradation.

## Reproducing these numbers

All five tools are CPU-only (or CPU-only except for one-time DINO encoding
of GT frames, cached after the first run) and live in `utils/`:
`diagnose_fov_occlusion_poisoning.py`, `diagnose_context_diversity_collapse.py`,
`diagnose_zero_overlap_fallback.py`, `diagnose_iou_estimation_noise.py`. The
drift check (#3) needs no script, just the `memory_corruption` column
already present in an existing `query_decomposition.csv` from
`analyze_retrieval_quality_decomposition.py`.
