# Memory Example Strips

CPU-only export, separate from the frozen metric-grid code. No new GPU job.
The existing five-column paper figure is left intact until real replacements
have been inspected. New figure labels call GeoCov **Ours**.

## Retrieval

```bash
python paper/make_memory_strips.py retrieval \
  --output "$HOME/memcam_results/retrieval_strips_$(date +%Y%m%d_%H%M%S)"
```

Defaults: 60-second manifest, baseline/FIFO-B32/GeoCov-B32, five examples,
one per trajectory, target stride 16, CPU scoring at width 256.
All three selected indices must come from the same logged section/query.
Output: **Target ground truth | Unbounded | FIFO | Ours**. No extra historical
ground truths, heatmaps, colored borders, or metric tables inside the strip.
Each example has labeled PNG/PDF, a label-free `_bare.png`, and JSON metadata.

`--content common` (default) loads all selected indices from the baseline video,
preserving the common-source control. These are policy-selected common-source
images, not the actual pixels that conditioned each independent rollout.
Use `--content own` to show the actual selected pixels from each policy video;
that comparison also includes differences in previously generated content.
State which version is shown in the figure caption.

To reformat existing examples without searching for new indices, add
`--cases /path/to/tables/selected_examples.csv`. FIFO is recovered from its
matching access trace, never invented. For 180-second cases, also supply the
appropriate `--manifest`, `--root`, and `--duration 180`.

Without `--cases`, candidate ranking uses target-match SSIM improvement over
Unbounded, with minimum 0.05 and no FIFO-win requirement. This measures effective
view/content match, not image corruption alone. Sidecars separately record
generated-to-target match, generated-to-own-index fidelity, and clean
historical-to-target view match. These selected extremes are not cohort results.

## Read/Reuse and Subsequent Deviation

Reuse the earlier CPU scan, without recomputing its full-video metrics:

```bash
python paper/make_memory_strips.py snowball \
  --cache "$HOME/memcam_results/snowball_20260909_100953" \
  --output "$HOME/memcam_results/snowball_strips_$(date +%Y%m%d_%H%M%S)"
```

The new search looks through the raw traces for an actual two-read chain:
memory conditions a section; a frame generated in that section is then
retrieved to condition a later section. The search extends up to six sections
after the first read. Both read overlaps must be at least 0.8 by default.

The screening rule uses two preceding sections as reference, requires at least
1 dB PSNR and 0.02 SSIM deterioration in both the first recipient section and
the two-section window beginning at reuse, and requires the first memory and
reused output to have lower PSNR than the pre-read reference. Ours need not win;
the video need not deteriorate monotonically. This is a NEW exploratory screen,
not retroactive success on the old eight-criterion test. Threshold differences
are recorded in `snowball_search.json`. Lower fidelity is not proof of error
propagation, object identity changes, or statistical significance.

Exported Unbounded strip: **Retrieved memory | Output retrieved again |
After reuse | Later**. Columns 1 and 2 are exact logged selected frames.
Column 3 is the second read's target, and column 4 is a cached frame near the
middle of the following section. Ours and ground truth at the identical indices
are exported as separate strips, not placed in a crowded combined figure.
The Ours strip is aligned output, not a reconstruction of its own read chain.
A separate PSNR/SSIM graph uses all cached sections; dashed/dotted lines mark
the two recipient sections. Cached scores inherit the old scan's source data
and sampling; the full source pixels are not reverified.

If no chain passes, output is named `inspection_only`, and JSON `selected` stays
empty. No unlinked episode is relabeled as snowballing. Inspect for the SAME
concrete visible error persisting or spreading along the chain before putting
an example in the paper. Even a confirmed visual recurrence is observational,
not a controlled proof that retrieval caused it.

This adapter supports MemCam's 76-frame sections, not WorldMem traces. WorldMem
needs its own recorded read identities/section mapping to establish an analogous
chain. Merely arranging worsening WorldMem frames would show drift, not establish
memory-mediated propagation.
