# MemCam Meeting Progress - 2026-08-28

## The headline

The project is not a B32-only result. The complete budget sweeps were already
run but were buried in the appendix and analysis artifacts. After promoting
them into the main paper, every tested Geometric Coverage budget beats
complete retention on the reported matched metric:

| System | Metric | B16 | B32 | B64 | B128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MemCam, 180s | LPIPS | 0.5876 | 0.5876 | **0.5865** | 0.5913 |
| MemCam, 180s | FVD | **446.1** | 476.6 | 493.9 | 515.5 |
| WorldMem, 60s | LPIPS | **0.525** | 0.534 | 0.545 | 0.577 |

The best capacity differs by metric, but Geometric Coverage does not collapse
outside B32. MemCam LPIPS changes by only 0.0048 across an eight-fold budget
range. This directly answers the strongest budget-overfitting criticism.

One protocol audit remains before putting unbounded in this same table: the
older budget-sweep artifact stores unbounded as 0.6004 LPIPS / 797.5 FVD,
while the later headline export stores 0.5980 / 734.2. Every GeoCov budget is
numerically better than both references, but the meeting should not call the
cross-row comparison matched until the metric configurations and video IDs
are checked.

## Mechanism evidence already completed

1. Across 4,200 unbounded queries, the selection gap grows with history. Mean
   late-minus-early change is +0.0723 with trajectory-bootstrap 95% CI
   [0.0396, 0.1088]; 13/15 trajectories have a positive trend.
2. The hindsight-best candidate remains approximately flat (+0.0008), while
   the candidate chosen by the real retriever worsens (+0.0732).
3. View alignment improves late (-0.0335), while selected-memory corruption
   worsens (+0.0873). The problem is not mainly wrong camera direction; it is
   plausible-view retrieval backed by damaged generated content.
4. In the common-source control, every selector reads pixels from the same
   unbounded rollout. Geometric Coverage still selects indices that are
   +4.629 dB PSNR and +0.1512 SSIM cleaner than unbounded, winning 15/15
   trajectories on both metrics.
5. On each policy's own rollout, Geometric Coverage-selected memories are
   +4.819 dB PSNR and +0.1670 SSIM cleaner than unbounded. Following-chunk
   SSIM improves by +0.0169 with CI [0.0012, 0.0392], while the PSNR interval
   crosses zero.
6. FIFO-32 is worse than unbounded on LPIPS, so an arbitrary small recent bank
   is not enough. Structured candidate-set composition matters.
7. The fixed-history cardinality intervention is now complete. Expanding the
   same recent B32 core to full history changes the selected identity 72.2% of
   the time, but fidelity is slightly higher rather than lower (+0.228 dB
   PSNR, +0.0011 SSIM). Candidate count alone does not explain corruption.
8. The common-source budget sweep is complete on fourteen matched
   trajectories. Every RI and GeoCov budget increase reduces retention gap but
   increases selection gap. GeoCov total diagnostic gap worsens from 0.1750
   at B16 to 0.2100 at B128, while unbounded is 0.2199. More capacity preserves
   options but makes the unchanged retriever less effective at using them.

## New work completed since the review

- The full MemCam and WorldMem budget curves are now in the main paper.
- The complexity statement now applies specifically to MemCam's implemented
  exhaustive FOV scan. The paper explicitly acknowledges ANN prefiltering and
  no longer presents exhaustive search as a universal lower bound.
- A fixed-history nested-pool intervention is completed. It
  freezes the generated video, target query, poses, candidate pixels, and one
  full vector of real FOV-overlap scores. It changes only which candidates are
  admitted: B32, B64, B128, B256, B512, B1024, and full history. Retrieval
  identity becomes unstable, but selected-frame fidelity does not decline.
- The multi-case ground-truth memory-cleaning replay has a single sequential,
  resume-safe H100 job. The harmful 90-second CUDA timeout was removed.
- The VBench-Long job now prints `nvidia-smi` and diagnoses the exact MoviePy
  compatibility requirement before running.

## Experiments that are genuinely left

### Acceptance-critical

1. **Multi-case GT-memory cleaning replay.** H100 generation. Keep selected
   identities and all preceding history fixed; replace only selected memory
   pixels with exact-index GT at the intervention section. This is the causal
   test of corruption propagation.
2. **Uniform reservoir-32 rollout.** H100 generation. This is the neutral
   capacity control missing from FIFO. It tests whether random cardinality
   reduction is sufficient or geometric composition is necessary.
3. **Headline uncertainty.** Reuse completed outputs. Report paired
   trajectory bootstrap intervals and win counts for LPIPS, video-resampling
   sensitivity for FVD, and video-bootstrap intervals for VBench.
4. **VBench-Long.** Fix the MoviePy environment, smoke-test one completed run,
   then score the locked matched policies. No videos need regeneration.
5. **Budget-protocol audit.** Compare video IDs and metric configuration for
   the two unbounded exports, then reevaluate baseline and the four GeoCov
   budgets together only if they differ. No videos need regeneration.

### Important reviewer insurance

6. **Indexed retrieval control.** Use a pose ANN index only as a prefilter,
   then run the unchanged FOV scorer on top-K. Report latency, recall of the
   exhaustive winner, and selected-memory fidelity. This concedes that ANN can
   address lookup cost while testing whether archive quality remains a
   separate problem.
7. **Untouched test trajectories.** Either generate a genuinely unused set
   after freezing the policy or keep the paper explicitly exploratory. The
   current fifteen MemCam trajectories were used during development.
8. **WorldMem final metric matrix.** Complete matched VBench and VBench-Long.
   Report CUT3R only if its GT sanity test is repaired.

### Not required for the core claim

- Another RI/GeoCov blend, quality gate, or hysteresis policy.
- CUT3R numbers from the currently invalid evaluator.
- A claim that Geometric Coverage is a new SLAM algorithm.
- A claim that one budget or one kernel is universally optimal.

## What to show in the meeting

1. The full budget table above. State plainly that the earlier draft hid the
   evidence and made the result look B32-specific.
2. The pool-growth decomposition: best available stays flat, selected gets
   worse, and the increase comes from stored-content corruption rather than
   view mismatch.
3. The common-source control with the +4.629 dB / +0.1512 result and 15/15
   trajectory wins.
4. The fixed-history result: winner identity changes from 0% to 72.2%, while
   PSNR/SSIM remain flat. State the negative conclusion clearly: cardinality
   alone is not the poisoning mechanism.
5. The all-budget retention-selection figure. Follow each family from B16 to
   larger budgets: every step moves left and upward. State that capacity buys
   retention at the cost of retrievability.
6. The causal cleaning replay diagram: same identity and history, generated
   memory pixels versus GT memory pixels, compare the next chunk.
7. A reviewer-response table separating completed rebuttals from jobs now in
   the queue. Do not describe missing experiments as completed results.

## Defensible current claim

Complete retention is not a safe empirical upper bound for a fixed-read
generative memory. Structured bounded memories outperform it across all tested
GeoCov budgets in two memory systems, and candidate-set composition changes
the fidelity of evidence chosen by an unchanged retriever. The fixed-history
intervention does not support a cardinality-only explanation. The remaining GT-cleaning
replay determines whether selected-memory corruption causally propagates into
the next generated chunk.
