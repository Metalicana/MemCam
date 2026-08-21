# ICLR Paper Draft

> **Internal evidence warning (2026-08-21):** the no-reference quality gate
> failed held-out calibration, and the fixed 75/25 Geometric Coverage-RI blend
> has not been validated as the final method. The QGRC method and victory prose
> in `main.tex` are aspirational scaffolding, not supported conclusions. See
> [`../docs/iclr_reviewer_risk_register.md`](../docs/iclr_reviewer_risk_register.md)
> before editing or citing the manuscript.

## Manuscript

- `main.tex`: complete two-column submission draft.
- `references.bib`: bibliography.
- `figures/`: generated PDF and PNG figures.
- `make_figures.py`: reproduces the paper figures from the consolidated
  measurements.

The working title is **The Archive Is Not the Context: Diagnosing and Curating
Memory for Long-Horizon Video Generation**.

## Paper Logic

The manuscript now follows the intended method-paper arc:

1. Long-horizon generators usually improve retrieval while allowing the
   underlying archive to grow.
2. The growing archive has linear storage and quadratic cumulative exhaustive
   search, yet the generator still consumes a fixed-size retrieved context.
3. More candidates can also reduce quality. Pool-growth and common-source
   analyses show increasing exposure to corrupted generated memories.
4. The response under evaluation is bounded Geometric Coverage with a limited
   RI contribution and, only if it passes held-out validation, a
   pose-calibrated conditioning-consistency admission gate. Generic
   no-reference IQA has been rejected.
5. FIFO, RI, Geometric Coverage, K-center, density-balanced coverage, Marginal
   Coverage Eviction (MCE), Surprise Forcing, and simple RI/geometry blends are
   baselines or ablations, not the full method.

## Build

The local workstation currently has no LaTeX compiler. On a machine with a TeX
distribution:

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Regenerate figures with:

```bash
MPLBACKEND=Agg python make_figures.py
```

Before submission, replace the generic two-column preamble with the official
ICLR template released for the target year.

## End-State Draft

The main text was intentionally written in an intended completed-paper voice,
but its QGRC claims are now stale hypotheses. Unknown numerical values remain
visible LaTeX macros such as `\finalLPIPS`, `\finalFVD`, `\finalVBenchGain`, and
`\finalRetrievalSpeedup`; no value should be filled without an audited final
artifact. Method claims must be revised after the causal-consistency validator,
the blend comparison, and the GT-content replay are complete.

The generic quality-estimator pilot failed and must not be used to justify a
gate. The replacement hypothesis is pose-calibrated conditioning consistency,
with a predeclared held-out deployment test. It is not part of the method unless
that test returns `INJECT`.

## Evidence Status

The diagnostic sections are supported by completed experiments:

- complete 15-trajectory 180-second unbounded/FIFO/RI/Geometric Coverage comparison at
  budget 32;
- pool-growth and view/corruption decomposition;
- selected-memory PSNR/SSIM analysis;
- common-source index-selection control;
- alternative-mechanism negative tests;
- one context-swap replay, labeled as underpowered.

The manuscript includes dedicated result tables for:

- LPIPS and FVD;
- all six VBench dimensions used by the project;
- CUT3R rotation, translation, and camera-control scores;
- archive storage, retrieval latency, and peak VRAM;
- quality-gate, geometric-only, ungated 75/25 blend, and full QGRC ablations.

Only the completed LPIPS/FVD and diagnostic values are currently populated.
The VBench, CUT3R, efficiency, replay, and full-method cells remain visible
placeholders; no numbers were inferred from partial runs.

The following are unfinished or required future work, even where the current
aspirational prose reads as though they succeeded:

- multi-case ground-truth memory-cleaning replay;
- cross-backbone and cross-representation validation on WorldMem, VMem, and
  point-cloud memory;
- complete matched VBench and CUT3R tables;
- matched H100 latency/VRAM measurements for unbounded and the final method;
- the pose-calibrated conditioning-consistency decision on held-out trajectories;
- the 75/25 blend comparison and coefficient sensitivity analysis;
- confidence intervals for aggregate FVD;
- final QGRC rollouts that consistently improve on Geometric Coverage.

The paper is a structural draft, not a submission-ready empirical package. The
risk register controls which method and mechanism claims survive into the next
revision.
