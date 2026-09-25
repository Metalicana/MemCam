# Editable Memory Diagnostics

Run `python paper/export_memory_diagnostics_tikz.py` from the repository root.
The generated `.tex` panels contain their coordinates directly; they need no
CSV file at LaTeX compile time. They require `pgfplots` (compatibility 1.18).
Use `three_panel_figure.tex` as the complete replacement for the supplied figure.
It includes the original LPIPS/FVD panel and both new panels in one self-contained
block, with shared colors, 32mm plot heights, fixed 52mm canvases, aligned axis
baselines, and aligned captions. It needs no separate input files or custom
color/method macros. The default middle panel shows absolute mean distances.
`preview.tex` compiles this complete figure. The earlier `subfigures_bc.tex` is
only a fragment for manuscripts that manage their own matching panel containers.
Compile the manuscript from `paper`, or adjust the input paths.

## Panels

- `retention_selection_tradeoff.tex`: exactly the original full-landscape
  coordinates from `make_figures.py`, with RI omitted and GeoCov renamed
  KEEPSAKE (Ours). Every bounded point has a B16/B32/B64/B128 label and a
  budget-specific marker. The points remain from the same 180-second,
  13-trajectory analysis; none are replaced with 60-second values. These are
  rounded legacy estimates, with no newly inferred uncertainty.
- `retrieval_quality_comparison.tex`: absolute mean view mismatch and selected
  memory corruption for Unbounded versus KEEPSAKE B32, on 15 matched 60-second
  trajectories and 1,380 identical sampled reads per method. Smaller is better.
- `retrieval_deterioration_comparison.tex`: alternative version showing both
  methods' late-minus-early changes, comparing sections 1--5 with 19--23.
  Positive values mean deterioration. Smaller changes do not establish smaller
  absolute errors.

Both comparison panels score selected historical IDs using the *same baseline
generated pixels*. They do not measure the two methods' own generated archives.
Queries are averaged within sections, sections within trajectories, and
trajectories equally. Whiskers use 10,000 whole-trajectory bootstrap samples,
seed 0. All metrics and policies use the same resampling indices. Paired
contrasts, rather than overlapping marginal intervals, assess method differences.

The actual 60-second result is less selected-memory corruption but more clean-view
mismatch under KEEPSAKE: 0.40786 versus 0.48709 corruption, and 0.32537 versus
0.12560 view mismatch. The corruption change is numerically smaller too, but the
paired interval for that change includes zero. The figure does not claim an
improvement in both quantities or causal error prevention.

`landscape_points.csv`, `retrieval_statistics.csv`, `paired_contrasts.csv`, and
`provenance.json` preserve the coordinates, source identity, cohort, sections,
and statistical definitions. Use natural TikZ sizing, not `resizebox`, to
preserve the readable label sizes. Increase the subfigure width for a larger
plot; the panel width follows `\linewidth`.

The two retrieval versions are not the same statistic: `mean` shows error
levels averaged over the rollout, while `change` shows late minus early.
For example, Unbounded corruption is 0.48709 on the first and +0.14757 on
the second. To regenerate the complete figure with the temporal-change panel,
run `python paper/export_memory_diagnostics_tikz.py --middle-panel change`.
Its axis label, subcaption and parent caption all change together.
