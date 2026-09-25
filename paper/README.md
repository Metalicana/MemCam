# Keepsake Paper Workspace

## Where Things Live

| Location | Contents |
| --- | --- |
| [main.tex](main.tex), [references.bib](references.bib) | Manuscript source and bibliography |
| [tikz/](tikz/README.md) | Editable plots, complete three-panel figure, and compiled preview |
| [figures/](figures/README.md) | Exported figures, draw.io sources, and qualitative frame assets |
| [configs/](configs/README.md) | Figure settings, selected examples, and reported input summaries |
| [results/](results/README.md) | Imported metric and diagnostic snapshots, with original reports |
| `*.py` | Existing figure builders, analysis tools, and experiment entry points |

The Python scripts stay here so imports, Slurm jobs, and existing run commands
keep working. Model code and evaluation implementations have not moved.

## Figure Entry Points

Run these from the repository root in the existing plotting environment.

| Figure | Command |
| --- | --- |
| Complete editable three-panel figure | `python paper/export_memory_diagnostics_tikz.py` |
| B32 quality comparison | `python paper/plot_editorial_b32.py` |
| Full budget comparison | `python paper/plot_grouped_budget_metrics.py` |
| Retention/selection budget sweep | `python paper/plot_gap_budget_sweep.py` |

The TikZ exporter writes [three_panel_figure.tex](tikz/three_panel_figure.tex).
See [TikZ instructions](tikz/README.md) for compilation and the alternative
late-minus-early middle panel.

The method, motivation, and revisit builders still use their original script
names. Their defaults now read `configs/`; video-based builds also require the
local videos or frame assets named in those configurations. See each script's
`--help` before running a build on another machine.

## Manuscript Build

With a LaTeX installation:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

`main.pdf` is an existing compiled snapshot, not automatically rebuilt when
the source changes. The standalone TikZ preview is `tikz/preview.tex`.

## Organization Notes

The September 24 cleanup moved result directories into `results/`, JSON input
files into `configs/`, and prior figure backups into `figures/archive/`.
Filenames within those directories are unchanged. Update explicit old input
paths in personal commands; the scripts' defaults have already been updated.
Previously exported provenance retains the paths used at export time.

The former paper-status document is preserved verbatim in
[docs/archive/paper_status_2026-09-06.md](../docs/archive/paper_status_2026-09-06.md).
It is historical context, not the current results index. Experiment instructions
are indexed in [docs/README.md](../docs/README.md).
