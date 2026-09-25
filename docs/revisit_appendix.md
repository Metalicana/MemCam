# Editable Revisit Appendix

The new collection is separate from the main-paper examples in
`paper/figures/revisit_editable/`. Open
`paper/figures/revisit_appendix/revisit_appendix.drawio` and use the bottom page
tabs. There are 25 pages: 10 MemCam scenes and all 15 WorldMem trajectories.
Each page has nine individually editable, embedded original frames and native
text labels. No external image links are needed to open the document.

`appendix_shortlist.drawio` and `appendix_shortlist.pdf` contain a smaller
10-example selection (7 MemCam, 3 WorldMem), manually chosen after visual review
for inspectable scene structure. It includes mixed outcomes, including a WorldMem
case where Unbounded retains a more recognizable view. This shortlist is not a
representative performance sample. `shortlist.json` records every choice and its
page in the full collection; `shortlist_overview_01.jpg` and
`shortlist_overview_02.jpg` provide a quick visual index.

Rows: Unbounded, FIFO B32, KEEPSAKE (Ours, B32). Columns: First visit,
Intervening view, Revisit. All methods use exactly the same frame indices.
The original main-paper example triplets are not repeated.

## Selection and Limits

- MemCam: the longest saved near-pose return from each of the 10 eligible scenes
  in the existing `oracle30` revisit-events file. This covers 10 of the 15 source
  scenes, not a new exhaustive pose search. Ties use the earliest first/return
  indices. The middle is the arithmetic midpoint; departure is not verified.
  Endpoint pose errors are at most 0.25 m and 5 degrees. Times use 30 FPS.
- WorldMem: the lowest exported geometry rank for every trajectory, using its
  departure-verified middle frame. Endpoint errors are at most 0.75 block units
  and 15 degrees. Times use the 10-FPS trajectory clock, not 15-FPS playback.
  The exporter did not log actual post-retry dataset identity: these are
  requested-trajectory correspondences, not verified actual-source/GT matches.
- Neither selection uses policy quality or generated pixels. Mixed outcomes
  remain included; the collection does not estimate average performance.
- "First visit" denotes the earlier selected endpoint, not necessarily the
  earliest-ever visit. Near-pose returns are not identical views. These images
  illustrate within-policy generated appearance, not GT reconstruction.
- Full source frames retain their original aspect ratios and colors. There is
  no crop, exposure adjustment, synthesis or alignment warp. No object boxes
  are automatically asserted; they can be added as editable draw.io rectangles.

The existing WorldMem verifier checks all nine decoded frames per example
against the exported preview bundle, allowing only small RGB decoder rounding.
Per-example `.provenance.json` files preserve source hashes, indices, pose evidence,
clock mapping, checks and limitations. `index.csv` maps PDF page numbers to
examples and timestamps. `captions.txt` contains publication caveats for each.

## Rebuild

Run from the repository root; the config resolves the downloaded source bundles.
These scripts only read those bundles and do not change or run any experiments.

```bash
python paper/build_revisit_appendix.py

/Applications/draw.io.app/Contents/MacOS/draw.io \
  --export --format pdf --all-pages --crop --disable-update \
  --output paper/figures/revisit_appendix/revisit_appendix.pdf \
  paper/figures/revisit_appendix/revisit_appendix.drawio

python paper/build_revisit_appendix.py --preview-only

/Applications/draw.io.app/Contents/MacOS/draw.io \
  --export --format pdf --all-pages --crop --disable-update \
  --output paper/figures/revisit_appendix/appendix_shortlist.pdf \
  paper/figures/revisit_appendix/appendix_shortlist.drawio
```

The last step requires Poppler's `pdftoppm` and renders the actual PDF export.
Full-page PNGs are in `previews/`; `overview_01.jpg` through `overview_05.jpg`
are browsing sheets. Individual editable documents are in `examples/`.

The master draw.io is approximately 70 MB because all 225 full-resolution images
are embedded. Individual documents are smaller and can be edited separately.
Generated assets are ignored by Git; transfer them explicitly when needed.

For a specific PDF page in LaTeX, after placing the exported PDF in `figures/`:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[page=1,width=\linewidth]{figures/revisit_appendix.pdf}
  \caption{Near-pose return in AncientTempleEnv\_5 (MemCam, 180 s).
  Rows show Unbounded, FIFO B32, and KEEPSAKE B32 at identical times.
  Endpoints are near-matched camera poses; the middle is an intervening view.}
  \label{fig:appendix-memcam-revisit-01}
\end{figure}
```

Use the corresponding caption and caveats for whichever page is included;
WorldMem's requested-trajectory limitation must not be relabeled as verified GT.
