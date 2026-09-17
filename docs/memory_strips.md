# Memory Example Strips

CPU-only export, separate from the frozen metric-grid code. No new GPU job.
The existing five-column paper figure is left intact until real replacements
have been inspected. New figure labels call GeoCov **Ours**.

## Same Spot on Repeated Visits

This is the appropriate figure for inspecting how a place changes on successive
returns, as opposed to searching for a memory read followed by a quality drop:

```bash
python paper/make_revisit_strips.py \
  --output "$HOME/memcam_results/revisit_strips_$(date +%Y%m%d_%H%M%S)"
```

Each figure has one GT reference at left and two rows (Unbounded, Ours), with
columns for three or four distinct visits. Visits are matched using absolute
camera position (0.25 m) and full orientation (5 degrees) relative to ONE anchor,
not transitive pose clusters. The camera must have left beyond twice that
tolerance for at least a second, and selected visits must be at least three
seconds apart. The supplied frame zero is excluded from generated-image visits.
All selected GT pairs must have SSIM >=0.9 at width 256, and the pose JSON's
frame keys must agree with the manifest's GT indexing. This last check fails
explicitly rather than silently shifting the GT reference.

Selection is based only on poses, GT agreement, visit count, and elapsed time;
it does not require Unbounded to degrade or Ours to win. One group per trajectory
is considered for the five exported figures. The main figure contains no metric
overlays or repeated GT tiles. A separate GT strip allows alignment inspection;
separate PSNR/SSIM plots compare each generated frame to its own exact-index GT.
Bare Unbounded/Ours strips and frame/provenance JSON are also saved.

No qualifying repeated view produces no fallback figure. `--min-visits 2`
explicitly requests pairs instead; `--duration`, `--manifest`, and `--root` can
select the longer 180-second rollouts. Do not call these causal snowball proofs:
they demonstrate how generated content behaves on repeated requested views.
The first repeated location can already be wrong relative to GT.

If the search returns no figures, audit availability before changing thresholds:

```bash
python paper/make_revisit_strips.py --diagnose --pose-stride 5 \
  --output "$HOME/memcam_results/revisit_diagnostics_$(date +%Y%m%d_%H%M%S)"
```

This counts separate returns including two-visit pairs, and samples GT agreement
for pairs versus groups of three or more. It reports the original pose tolerance
and explicitly separate 0.5m/10deg and 1m/15deg sensitivity profiles. Each profile
still requires a sustained departure beyond twice its own pose tolerance.
Profiles need not have monotonically increasing visit counts because a wider
tolerance can merge continuous visits. Up to 12 groups per category/profile
are checked, spread across pose-ranked candidates; best checked SSIM is not an
exhaustive maximum. GT-only previews show why candidates pass or fail. No
generated MP4 is decoded, no policy-quality filtering is performed, and rejected
groups are not promoted to same-view examples. Alternative anchors are now
deduplicated only when their selected frame lists are identical, not merely
when visits fall into the same coarse time bins.

To render the saved GT previews for visual comparison without applying the
GT-SSIM gate again:

```bash
python paper/make_revisit_strips.py \
  --from-diagnostics "$HOME/memcam_results/revisit_diagnostics_20260917_105624" \
  --profile strict --top 15 \
  --output "$HOME/memcam_results/revisit_figures_$(date +%Y%m%d_%H%M%S)"
```

This decodes the exact preview indices, with no new pose search. By default it
uses the saved three-or-more-visit preview when available, otherwise the saved
two-visit preview. Use `--visit-category 2visits` or `3plus` to choose explicitly,
and `--rows 28,53,68` to restrict trajectories. Manifest, video root, duration,
and policy folder are inherited from the diagnostic report unless overridden.
GT agreement scores and bypassed-gate provenance remain in the JSON; the
separate GT strip supports visual inspection. These are inspection candidates,
not automatically verified identical views or evidence of causal snowballing.

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
