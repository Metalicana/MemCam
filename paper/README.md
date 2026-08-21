# ICLR Paper Draft

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
4. The proposed response is **Quality-Gated Geometric-Rarity Curation
   (QGRC)**: reject low-quality incoming observations using a calibrated
   no-reference estimator, then retain the top-$B$ items under
   `0.75 * normalized Geometric Coverage + 0.25 * normalized RI`.
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

The main text is intentionally written in the intended completed-paper voice:
QGRC wins the fixed-budget quality/efficiency comparison, and the discussion
explains why. Unknown numerical values are still represented by visible LaTeX
macros such as `\finalLPIPS`, `\finalFVD`, `\finalVBenchGain`, and
`\finalRetrievalSpeedup`; no result has been invented. Replace those macros
only from audited final artifacts. The prose states the intended end result so
the remaining experiments have an explicit falsifiable target.

The quality estimator and threshold must be selected on calibration
trajectories disjoint from the final 15-video benchmark. The current
baseline-versus-Geometric-Coverage interactive estimator run is a pilot, not
the final disjoint calibration.

## Evidence Status

The draft claims only results currently supported by completed experiments:

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

The following are correctly presented as unfinished or required future work:

- multi-case ground-truth memory-cleaning replay;
- cross-backbone and cross-representation validation on WorldMem, VMem, and
  point-cloud memory;
- complete matched VBench and CUT3R tables;
- matched H100 latency/VRAM measurements for unbounded and the final method;
- the quality-estimator winner and locked threshold on a disjoint calibration set;
- the gate-first 75/25 QGRC implementation and its component ablations;
- confidence intervals for aggregate FVD;
- final QGRC rollouts that consistently improve on Geometric Coverage.

The paper is therefore a complete, honest draft, not yet a submission-ready
empirical package.
