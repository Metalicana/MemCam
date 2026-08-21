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
4. The proposed response is **Quality-Gated Geometric-Rarity Curation**: reject
   unreliable supported observations, protect rare content, and use the
   remaining capacity for Geometric Coverage.
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
- gate/rarity/geometric component ablations.

Only the completed LPIPS/FVD and diagnostic values are currently populated.
The VBench, CUT3R, efficiency, and full-method cells are marked `TBD`; no
numbers were inferred from partial runs.

The following are correctly presented as unfinished or required future work:

- multi-case ground-truth memory-cleaning replay;
- cross-backbone and cross-representation validation on WorldMem, VMem, and
  point-cloud memory;
- complete matched VBench and CUT3R tables;
- matched H100 latency/VRAM measurements for unbounded and the final method;
- the quality-gated geometric-rarity implementation and its component ablations;
- confidence intervals for aggregate FVD;
- a final policy that consistently improves on Geometric Coverage rather than
  an uncalibrated RI/geometry blend.

The paper is therefore a complete, honest draft, not yet a submission-ready
empirical package.
