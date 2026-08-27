# Paper Status

Working title:

> **The Archive Is Not the Context: Diagnosing and Curating Memory for
> Long-Horizon Video Generation**

The manuscript is now written as a diagnostic and controlled-intervention
paper. It does not claim a quality gate, Geometric-RI blend, or QGRC full
method. Geometric Coverage is the strongest tested retention intervention and
is explicitly described as an adaptation of SLAM-style keyframe redundancy.

## Paper spine

1. The archive grows, but the generator reads a fixed-size context.
2. Candidate competition can make complete retention statistically harmful.
3. Retention quality and retrievability are different objectives.
4. Unbounded MemCam increasingly selects corrupted generated content even as
   selected camera views become slightly better aligned.
5. A common-source control shows that structured curation selects cleaner
   indices under identical historical pixels.
6. Geometric coverage is the strongest tested online criterion; rarity is
   complementary but weaker, and reliability is poorly observable at write
   time.
7. Matched WorldMem results reproduce the broad advantage of structured
   bounded memory under a latent-memory interface.

## Supported headline results

Matched MemCam, 180 seconds, 15 trajectories:

| Policy | Stored frames | LPIPS | FVD |
| --- | ---: | ---: | ---: |
| Unbounded | 5,397 | 0.5980 | 734.2 |
| FIFO-32 | 32 | 0.6514 | 677.3 |
| RI-32 | 32 | 0.5939 | 550.4 |
| Geometric Coverage-32 | 32 | **0.5876** | **476.6** |

Common-source control relative to Unbounded:

- RI selects indices with `+1.775 dB` PSNR and `+0.0672` SSIM.
- Geometric Coverage selects indices with `+4.629 dB` PSNR and `+0.1512`
  SSIM, winning both metrics on 15/15 trajectories.

Matched WorldMem, first 15 videos, 60 seconds, B32:

| Policy | LPIPS | FVD |
| --- | ---: | ---: |
| Unbounded | 0.652 | 3077.6 |
| FIFO-32 | 0.689 | 3554.9 |
| Latent-RI-32 | 0.546 | 1160.4 |
| Geometric Coverage-32 | **0.534** | **1116.9** |

## Explicitly unsupported claims

- Candidate-pool growth alone causally produces the observed degradation.
- Archive growth directly dilutes MemCam denoiser attention.
- Corrupted selected memories fully explain downstream FVD.
- Generic IQA or pose-conditioned consistency provides a deployable gate.
- Geometric Coverage is a novel SLAM algorithm or globally optimal coverage
  objective.
- One concrete RGB-frame policy transfers unchanged to every representation.
- CUT3R camera scores are valid under the current evaluator.

## Remaining high-value experiments

1. Complete the multi-case ground-truth content-cleaning replay.
2. Run privileged `Oracle-clean`, `Oracle-future`, and `Oracle-both` policies
   at B32 to measure headroom over Geometric Coverage.
3. Finish the locked WorldMem metric matrix, especially standard VBench.
4. Add a true SLAM keyframe-culling implementation if a reviewer-facing
   baseline can be matched without changing the retriever.

These are future additions, not manuscript placeholders. The current paper
contains no fabricated result macros or empty metric tables.

## Figures

Regenerate all manuscript figures with:

```bash
python paper/make_figures.py
```

Generated artifacts are stored under `paper/figures/` in both PDF and PNG
formats. `model_architecture` now depicts the actual fixed-budget Geometric
Coverage controller.

## Build

The manuscript uses standard LaTeX plus `natbib`:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The local development machine used for the latest rewrite did not have a
LaTeX engine installed, so the final PDF must be compiled on a machine with a
TeX distribution or in Overleaf.
