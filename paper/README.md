# Paper Status

Working title:

> **GeoCov: Geometry-Aware Fixed-Budget Memory for Long-Horizon Video
> Generation**

## Paper spine

1. A useful video memory must preserve earlier views and supply them when the
   camera returns. Retention and retrieval are separate requirements.
2. History retrieval and selective context construction address these
   requirements differently. Complete retention leaves selection unresolved;
   tested recency and appearance rules lose useful history or leave retained
   evidence unused. Figure 1 shows these complementary failures.
3. The design principle is view coverage and substitutability: retain views
   with little support, and remove observations with geometric and visual
   substitutes. GeoCov implements this with a fixed archive budget.
4. Matched MemCam and WorldMem results test the principle in two systems;
   the generator and retriever are unchanged within each comparison.
5. Shared-pixel and fixed-history controls support the role of candidate
   composition. Random pool expansion does not establish candidate-count harm.

## Literature grounding

- History retrieval: [Context-as-Memory](https://arxiv.org/html/2506.03141v1),
  Section 3.3, and [MemCam](https://arxiv.org/html/2603.26193v1), Section 3.2,
  retrieve camera-relevant frames from historical sequences.
- Selective context construction: [FramePack](https://arxiv.org/abs/2504.12626)
  compresses context according to frame importance, including time and feature
  similarity. [MemFlow](https://arxiv.org/html/2512.14699v1), Section 3.2,
  combines semantic retrieval with a first-frame prototype of the prior chunk.
- These are design tendencies, not mutually exclusive system classes. A
  compact conditioning context does not imply a bounded persistent archive.
- FIFO and RI demonstrate the failure modes inside our system; they are not
  reproductions of FramePack or MemFlow. Do not attribute the measured gaps to
  those external systems.
- The temporal corruption analysis supports a concern about reusing generated
  errors. It does not establish a causal snowballing effect from archive growth.

## Comparison protocol

- Complete retention, GeoCov, and RI contain the initial conditioning frame.
- GeoCov and RI explicitly reserve that frame; complete retention keeps it by
  construction. FIFO and K-center do not reserve it.
- All bounded MemCam policies temporarily protect the latest section endpoint
  during an update, and protected items count toward the stated budget.
- The main text reports the initial-frame sensitivity analysis rather than
  presenting the common-source gain as an undifferentiated mechanism result.

## Headline results

Matched MemCam, 180 seconds, 15 trajectories:

| Policy | Stored frames | LPIPS | FVD |
| --- | ---: | ---: | ---: |
| Complete retention | 5,397 | 0.5980 | 734.2 |
| FIFO-32 | 32 | 0.6514 | 677.3 |
| RI-32 | 32 | 0.5939 | 550.4 |
| GeoCov-32 | 32 | **0.5876** | **476.6** |

Matched WorldMem, first 15 videos, 60 seconds, B32:

| Policy | LPIPS | FVD |
| --- | ---: | ---: |
| Complete retention | 0.652 | 3077.6 |
| FIFO-32 | 0.689 | 3554.9 |
| Latent-RI-32 | 0.546 | 1160.4 |
| Geometric Coverage-32 | **0.534** | **1116.9** |

Common-source control relative to complete retention:

- RI: `+1.775 dB` PSNR and `+0.0672` SSIM.
- GeoCov: `+4.629 dB` PSNR and `+0.1512` SSIM, positive on 15/15
  trajectories.

Fixed-history full pool relative to the shared B32 core:

- Retrieval identity changes on `72.2%` of queries.
- PSNR changes by `+0.228 dB`, 95% CI `[-0.109, +0.664]`.
- SSIM changes by `+0.0011`, 95% CI `[-0.0117, +0.0188]`.
- Decision: `DOES_NOT_SUPPORT_CARDINALITY_HARM`.

## Unsupported claims

- Candidate count alone degrades selected-memory fidelity.
- Archive growth directly dilutes denoiser attention.
- GeoCov detects corrupted images online.
- Cleaner selected memories fully cause the downstream FVD improvement.
- GeoCov is a novel SLAM algorithm or universally optimal policy.
- CUT3R and VBench-Long are complete and valid.
- End-to-end latency and peak-memory gains have already been measured.

## Follow-up evidence

These are opportunities to strengthen the existing contribution. The current
writing priority is the design argument above, not expanding the method.

1. Run GeoCov without initial-frame reservation and FIFO with the same
   reservation.
2. Add a uniform reservoir/random-B32 closed-loop capacity baseline.
3. Audit the older unbounded budget-sweep artifact against the headline run.
4. Add uncertainty for headline LPIPS, FVD, and VBench comparisons.
5. Measure end-to-end latency and peak memory for complete retention and
   GeoCov under matched hardware and video lengths.
6. Evaluate frozen constants on untouched trajectories.
7. Finish the matched WorldMem VBench/VBench-Long matrix.
8. Repair CUT3R ground-truth sanity before reporting camera metrics.
9. Run the multi-trajectory exact-index content-replacement replay if making a
   causal propagation claim.

## Figures and build

Regenerate the scripted analytical figures with:

```bash
python paper/make_figures.py
```

The qualitative common-source and eviction figures are stored directly in
`paper/figures/` from their corresponding analysis outputs.

Build with:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The current layout was compiled with Tectonic 0.17.0 and visually checked in
`main.pdf`. The main paper uses two columns; the appendix uses full-width
tables, with large qualitative figures grouped after the quantitative sections.
