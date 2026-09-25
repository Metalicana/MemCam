# Figure Assets

## Current Entry Points

| Figure | Editable Source / Export |
| --- | --- |
| Three-panel quantitative figure | [../tikz/three_panel_figure.tex](../tikz/three_panel_figure.tex), [preview](../tikz/preview.pdf) |
| Method | `ICLR27 Method Figure.drawio.xml`, `ICLR27_Method_real_frames.pdf` |
| Motivation | `ICLR27_Motivation_two_row.drawio`, matching PDF/PNG |
| B32 and budget metrics | `metric_bars/` |
| Editable revisit examples | `revisit_editable/` |
| Appendix revisit collection | `revisit_appendix/revisit_appendix.drawio`, `revisit_appendix/index.csv` |
| 180-second GT/rollout comparisons | `rollout_comparisons_180s/` |
| Retention/selection landscape | `retention_selection_policy_landscape.pdf` |
| Individual retrieval diagnostics | `retrieval_deterioration_panels/` |

Keep image assets and provenance sidecars with their figure. Most generated
assets are intentionally ignored by Git and may exist only in this checkout;
already tracked paper assets remain tracked.

## Prior Versions

`archive/` holds the existing `method_previous`, `motivation_previous`,
`motivation_before_right_crop`, `motivation_facade_backup`, and `teaser_previous`
directories. They were moved intact, not deleted. The method builder still
uses a template from `archive/method_previous/`.

Timestamped analysis and review directories remain in their original locations
because current builders and figure provenance reference them. Do not treat
those frame assets or analysis CSVs as disposable build caches.
