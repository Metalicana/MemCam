"""Export editable PGFPlots panels from the original landscape and matched reads."""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.paired_retrieval_curves import load_matched_sections
from paper.plot_retrieval_deterioration import FIELDS, IDENTITY

MARKS = {16: "*", 32: "square*", 64: "triangle*", 128: "diamond*"}
COLORS = {"FIFO": "D55E00", "KEEPSAKE (Ours)": "4C8C4A", "Unbounded": "4D4D4D"}
COMPARISON_RUNS = (("baseline", "Unbounded"), ("fifo_b32", "FIFO"),
                   ("slam_b32_covisibility", "KEEPSAKE (Ours)"))
COMPARISON_STYLES = {"Unbounded": ("diagUnbounded", "*", .18),
                     "FIFO": ("diagFifo", "triangle*", 0.),
                     "KEEPSAKE (Ours)": ("diagKeep", "square*", -.18)}


def original_points(path):
    """Read only literal coordinates, without importing/running the legacy plotter."""
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "plot_retention_selection_tradeoff")
    families = next(n.value for n in function.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "families" for t in n.targets))
    points = []
    for key, value in zip(families.keys, families.values):
        family = ast.literal_eval(key)
        if family not in ("FIFO", "GeoCov"):
            continue
        triples = next(ast.literal_eval(v) for k, v in zip(value.keys, value.values)
                       if ast.literal_eval(k) == "points")
        if [p[0] for p in triples] != list(MARKS):
            raise ValueError("Expected all four original budgets")
        for budget, retention, selection in triples:
            points.append(dict(policy="KEEPSAKE (Ours)" if family == "GeoCov" else family,
                               budget=budget, retention_gap=retention, selection_gap=selection))
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "scatter"
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "full_ax"]
    if len(points) != 8 or len(calls) != 1:
        raise ValueError("Unexpected original landscape structure")
    x, y = [ast.literal_eval(v) for v in calls[0].args[:2]]
    points.append(dict(policy="Unbounded", budget=None, retention_gap=x, selection_gap=y))
    if x != 0 or not np.isfinite([[p["retention_gap"], p["selection_gap"]] for p in points]).all():
        raise ValueError("Invalid original coordinates")
    return points


def statistics(path, duration=60, videos=15, bootstrap=10000, seed=0):
    if bootstrap < 100:
        raise ValueError("Need at least 100 bootstrap draws")
    loaded = load_matched_sections(path, COMPARISON_RUNS, duration, videos)
    values = []
    for run, _ in COMPARISON_RUNS:
        identities, sections, array, _, count = loaded[run]
        values.append(array)
    values = np.stack(values)
    k = len(sections) // 4
    means = values.mean(axis=2)
    changes = values[:, :, -k:].mean(axis=2) - values[:, :, :k].mean(axis=2)
    draws = np.random.default_rng(seed).integers(0, videos, (bootstrap, videos))
    rows, contrasts = [], []
    for statistic, array in (("rollout_mean", means), ("late_minus_early", changes)):
        for policy_index, (_, policy) in enumerate(COMPARISON_RUNS):
            per_trajectory = array[policy_index]
            ci = np.quantile(per_trajectory[draws].mean(axis=1), [.025, .975], axis=0)
            for i, metric in enumerate(FIELDS):
                rows.append(dict(statistic=statistic, policy=policy, metric=metric,
                                 mean=float(per_trajectory[:, i].mean()),
                                 ci_low=float(ci[0, i]), ci_high=float(ci[1, i])))
        for policy_index, (_, policy) in enumerate(COMPARISON_RUNS[1:], 1):
            difference = array[policy_index] - array[0]
            ci = np.quantile(difference[draws].mean(axis=1), [.025, .975], axis=0)
            for i, metric in enumerate(FIELDS):
                contrasts.append(dict(statistic=statistic, policy=policy, reference="Unbounded",
                                      metric=metric, difference=float(difference[:, i].mean()),
                                      ci_low=float(ci[0, i]), ci_high=float(ci[1, i])))
    metadata = dict(source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    parameters=dict(duration=duration, expected_videos=videos, bootstrap=bootstrap,
                                    seed=seed, content_run="baseline"),
                    selectors=[dict(run=run, policy=policy, budget=None if run == "baseline" else 32)
                               for run, policy in COMPARISON_RUNS],
                    trajectories=[dict(zip(IDENTITY, i)) for i in identities], section_indices=sections,
                    queries_per_policy=count, early_sections=sections[:k], late_sections=sections[-k:],
                    exported_statistics=rows, paired_contrasts=contrasts,
                    method="Same queries and baseline source pixels for all three selectors. Query means "
                    "within sections, equal section weights within trajectories, equal trajectory weights. "
                    "Shared whole-trajectory bootstrap draws across policies, metrics and statistics.",
                    limitations="Not each policy's own generated memory pixels and not a causal replay. "
                    "Banks and selected IDs come from each policy's own trace. All initial-frame selections "
                    "are included. CSV checks do not independently validate features, video hashes or GT mapping.")
    return rows, metadata


PREFIX = r"""% Generated by paper/export_memory_diagnostics_tikz.py.
% Requires \usepackage{pgfplots} and \pgfplotsset{compat=1.18}.
% Input inside a subfigure; do not resizebox the result (fonts would shrink).
\begingroup
\definecolor{diagKeep}{HTML}{4C8C4A}
\definecolor{diagFifo}{HTML}{D55E00}
\definecolor{diagUnbounded}{HTML}{4D4D4D}
\begin{tikzpicture}
"""
SUFFIX = "\\end{axis}\n\\end{tikzpicture}\n\\endgroup\n"


def landscape_tex(points):
    text = PREFIX + r"""% Exact original 180-s / 13-trajectory rounded points, with RI omitted.
\begin{axis}[
  scale only axis, width=\dimexpr\linewidth-27pt\relax, height=36mm,
  xmin=-0.007, xmax=0.220, ymin=0.025, ymax=0.239,
  xtick={0,0.10,0.20}, ytick={0.05,0.10,0.15,0.20},
  scaled ticks=false,
  tick label style={font=\scriptsize,/pgf/number format/fixed,/pgf/number format/precision=2},
  label style={font=\small},
  xlabel={Retention gap $\downarrow$},
  axis lines=left, axis line style={black!60}, tick style={black!60},
  tick align=outside, grid=major, grid style={black!12,line width=0.25pt},
  clip=false, enlargelimits=false,
]
\node[anchor=south west,font=\small,inner sep=0pt]
  at (rel axis cs:0,1.04) {Selection gap $\downarrow$};
"""
    for policy, color in (("FIFO", "diagFifo"), ("KEEPSAKE (Ours)", "diagKeep")):
        group = [p for p in points if p["policy"] == policy]
        coords = " ".join(f"({p['retention_gap']:.4f},{p['selection_gap']:.4f})" for p in group)
        text += f"\\addplot[{color},line width=1pt,mark=none,->] coordinates {{{coords}}};\n"
        for p in group:
            x, y, b = p["retention_gap"], p["selection_gap"], p["budget"]
            text += (f"\\addplot[only marks,{color},mark={MARKS[b]},mark size=2.3pt,\n"
                     f"  mark options={{draw=white,line width=0.35pt,fill={color}}}] coordinates {{({x:.4f},{y:.4f})}};\n")
            # B32 and B64 are close: alternate label sides without moving the data.
            anchor, shift = ("east", -4) if (policy == "FIFO" and b == 16) or (policy != "FIFO" and b == 64) else ("west", 4)
            text += (f"\\node[anchor={anchor},font=\\scriptsize\\bfseries,text={color},\n"
                     f"  inner sep=0.7pt,fill=white,xshift={shift}pt]\n"
                     f"  at (axis cs:{x:.4f},{y:.4f}) {{B{b}}};\n")
    unbounded = next(p for p in points if p["policy"] == "Unbounded")
    text += (f"\\addplot[only marks,diagUnbounded,mark=*,mark size=2.6pt]\n"
             f"  coordinates {{(0,{unbounded['selection_gap']:.4f})}};\n"
             f"\\node[anchor=west,font=\\scriptsize\\bfseries,text=diagUnbounded,xshift=4pt,inner sep=0pt]\n"
             f"  at (axis cs:0,{unbounded['selection_gap']:.4f}) {{Unbounded}};\n")
    text += r"""\node[font=\scriptsize\bfseries,text=diagFifo,inner sep=1pt]
  at (axis cs:0.16,0.138) {FIFO};
\node[font=\scriptsize\bfseries,text=diagKeep,align=center,inner sep=1pt]
  at (axis cs:0.125,0.181) {KEEPSAKE\\(Ours)};
"""
    return text + SUFFIX


def comparison_tex(rows, changes=False):
    statistic = "late_minus_early" if changes else "rollout_mean"
    metrics = ("selected_view_mismatch", "selected_memory_corruption")
    selected = [r for r in rows if r["statistic"] == statistic and r["metric"] in metrics]
    expected = {(policy, metric) for policy in COMPARISON_STYLES for metric in metrics}
    if len(selected) != len(expected) or {(r["policy"], r["metric"]) for r in selected} != expected:
        raise ValueError("Need all three policies and both metrics exactly once")
    if changes:
        xmin = min(0, min(r["ci_low"] for r in selected)) - .025
        xmax = max(0, max(r["ci_high"] for r in selected)) + .025
        label = r"Late $-$ early distance"
    else:
        xmin, xmax = 0., max(r["ci_high"] for r in selected) + .035
        label = r"Feature distance $\downarrow$"
    text = PREFIX + rf"""% Matched 60-s / 15-trajectory comparison; common baseline pixels.
\begin{{axis}}[
  scale only axis, width=\dimexpr\linewidth-17pt\relax, height=36mm,
  xmin={xmin:.6f}, xmax={xmax:.6f}, ymin=-0.28, ymax=1.56,
  ytick=\empty, xtick distance={.1 if changes else .2}, scaled ticks=false,
  tick label style={{font=\scriptsize}}, label style={{font=\small}},
  xlabel={{{label}}}, axis lines=left,
  y axis line style={{draw=none}}, axis line style={{black!60}}, tick style={{black!60}},
  tick align=outside, xmajorgrids=true, grid style={{black!12,line width=0.25pt}},
  clip=false, enlargelimits=false,
  legend style={{font=\scriptsize,draw=none,fill=none,at={{(0,1.01)}},anchor=south west,
                legend columns=1,inner sep=0pt,row sep=-1pt}},
  legend cell align=left,
]
\addlegendimage{{only marks,mark=*,diagUnbounded}}
\addlegendentry{{Unbounded}}
\addlegendimage{{only marks,mark=triangle*,diagFifo}}
\addlegendentry{{FIFO}}
\addlegendimage{{only marks,mark=square*,diagKeep}}
\addlegendentry{{KEEPSAKE (Ours)}}
\node[anchor=west,font=\scriptsize,inner sep=1pt,fill=white]
  at (axis cs:{xmin:.6f},1.40) {{View mismatch}};
\node[anchor=west,font=\scriptsize,inner sep=1pt,fill=white]
  at (axis cs:{xmin:.6f},0.40) {{Memory corruption}};
"""
    if changes:
        text += r"\draw[black!55,dashed,line width=0.5pt] (axis cs:0,-0.23) -- (axis cs:0,1.3);" + "\n"
    for row in selected:
        color, marker, offset = COMPARISON_STYLES[row["policy"]]
        y = 1 - metrics.index(row["metric"]) + offset
        x, lo, hi = (row[k] for k in ("mean", "ci_low", "ci_high"))
        text += (f"\\draw[{color},line width=1.1pt] (axis cs:{lo:.6f},{y:.2f}) -- (axis cs:{hi:.6f},{y:.2f});\n"
                 f"\\addplot[only marks,{color},mark={marker},mark size=2.4pt]\n"
                 f"  coordinates {{({x:.6f},{y:.2f})}};\n")
    return text + SUFFIX


def long_horizon_tex():
    # Coordinates supplied in the user's figure; this is a layout-only rewrite.
    return PREFIX + r"""\begin{axis}[
  keepsake panel axes,
  xmin=0.58, xmax=0.665, ymin=430, ymax=850,
  xtick={0.58,0.62,0.66}, ytick={450,600,750}, scaled ticks=false,
  xlabel={LPIPS $\downarrow$},
  tick label style={font=\scriptsize}, label style={font=\small},
  axis lines=left, axis line style={black!60}, tick style={black!60},
  tick align=outside, grid=major, grid style={black!12,line width=0.25pt},
  clip=false, enlargelimits=false,
]
\node[anchor=south west,font=\small,inner sep=0pt]
  at (rel axis cs:0,1.04) {FVD $\downarrow$};
\addplot[only marks,mark=*,mark size=2.6pt,diagUnbounded]
  coordinates {(0.5980,734.2)};
\node[anchor=south west,align=left,font=\scriptsize,text=diagUnbounded,
      inner sep=0pt,xshift=3pt,yshift=3pt]
  at (axis cs:0.5980,734.2) {MemCam\\[-1pt](0.5980, 734.2)};
\addplot[only marks,mark=square*,mark size=2.4pt,diagFifo]
  coordinates {(0.6514,677.3)};
\node[anchor=north east,align=right,font=\scriptsize,text=diagFifo,
      inner sep=0pt,xshift=-3pt,yshift=-4pt]
  at (axis cs:0.6514,677.3) {FIFO\\[-1pt](0.6514, 677.3)};
\addplot[only marks,mark=diamond*,mark size=2.8pt,diagKeep]
  coordinates {(0.5876,476.6)};
\node[anchor=south west,align=left,font=\scriptsize,text=diagKeep,
      inner sep=0pt,xshift=4pt,yshift=4pt]
  at (axis cs:0.5876,476.6) {KEEPSAKE (Ours)\\[-1pt](0.5876, 476.6)};
""" + SUFFIX


def fixed_canvas(tex):
    """Fit our generated panels into the same canvas without scaling their fonts."""
    tex = tex.replace(r"\begin{tikzpicture}",
                      "\\begin{tikzpicture}\n"
                      "\\path[use as bounding box] (0,0) rectangle (\\linewidth,52mm);")
    for margin in (17, 27):
        tex = tex.replace(r"scale only axis, width=\dimexpr\linewidth-" + str(margin)
                          + r"pt\relax, height=36mm,", "keepsake panel axes,")
    if tex.count("keepsake panel axes,") != 1:
        raise ValueError("Panel must use exactly one shared axis geometry")
    return tex


def three_panel_tex(points, rows, changes=False):
    panels = ((long_horizon_tex(), "Long-horizon quality.", "fig:long-horizon-extension"),
              (comparison_tex(rows, changes), "Memory-quality change." if changes else "Retrieved-memory quality.",
               "fig:retrieval-quality-paired"),
              (landscape_tex(points), "Retention--selection tradeoff.", "fig:retention-selection-tikz"))
    text = r"""% Complete replacement for the supplied figure, not an extra nested figure.
% Preamble: \usepackage{pgfplots,subcaption}
%           \pgfplotsset{compat=1.18}
% Self-contained: no external inputs, image files, colors or \Keep macro required.
\begin{figure}[t]
\centering
\pgfplotsset{keepsake panel axes/.style={
  scale only axis, at={(24pt,10mm)}, anchor=south west,
  width=\dimexpr\linewidth-30pt\relax, height=32mm
}}
\captionsetup[subfigure]{font=small,skip=4pt,justification=centering,singlelinecheck=false}
"""
    for index, (panel, caption, label) in enumerate(panels):
        text += ("\\begin{subfigure}[t]{0.32\\linewidth}\n"
                 "\\centering\\vspace{0pt}\n" + fixed_canvas(panel)
                 + f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{subfigure}}")
        text += "\\hfill%\n" if index < 2 else "\n"
    description = ("Late-minus-early changes in view mismatch and memory corruption, "
                   "comparing sections 1--5 with 19--23" if changes else
                   "Mean view mismatch and memory corruption")
    text += r"""\caption{
\textbf{Generation quality and memory diagnostics.}
(a) Fifteen matched 180-second MemCam rollouts; bounded methods use $B=32$.
""" + "(b) " + description + r""" on fifteen matched
60-second rollouts, using common baseline pixels and DINOv2 cosine distance;
FIFO and KEEPSAKE use $B=32$; whiskers show 95\% trajectory-bootstrap intervals.
(c) Original thirteen-trajectory, 180-second retention--selection analysis;
labels denote memory budgets. Lower is better on both axes.
}
\label{fig:three-panel-placeholder}
\end{figure}
"""
    return text


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export(legacy, queries, output, middle_panel="mean"):
    points = original_points(legacy)
    rows, metadata = statistics(queries)
    output.mkdir(parents=True, exist_ok=True)
    (output / "retention_selection_tradeoff.tex").write_text(landscape_tex(points))
    (output / "retrieval_quality_comparison.tex").write_text(comparison_tex(rows))
    (output / "retrieval_deterioration_comparison.tex").write_text(comparison_tex(rows, changes=True))
    (output / "three_panel_figure.tex").write_text(three_panel_tex(points, rows, changes=middle_panel == "change"))
    write_csv(output / "landscape_points.csv", points)
    write_csv(output / "retrieval_statistics.csv", rows)
    write_csv(output / "paired_contrasts.csv", metadata["paired_contrasts"])
    metadata.update(landscape=dict(duration_sec=180, trajectories=13,
                    source=str(legacy.resolve()), source_kind="original_rounded_figure_coordinates",
                    sha256=hashlib.sha256(legacy.read_bytes()).hexdigest(), points=points,
                    limits="Original coordinates, not recomputed raw results; no inferred error bars. RI omitted from this presentation only."),
                    note="Landscape (180 s, n=13) and retrieval comparison (60 s, n=15) are different diagnostics/cohorts.",
                    combined_figure=dict(middle_panel=middle_panel,
                        quality_points_source="User-supplied LaTeX, preserved without recomputing FVD/LPIPS",
                        canvas_height_mm=52, axis_height_mm=32))
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Wrote complete three-panel figure, individual panels and statistics to {output}")
    for row in metadata["paired_contrasts"]:
        if row["statistic"] == "rollout_mean" and row["metric"] in FIELDS[:2]:
            print(f"{row['metric']}: {row['policy']} - Unbounded = {row['difference']:+.6f} "
                  f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, default=ROOT / "paper/make_figures.py")
    parser.add_argument("--queries", type=Path,
                        default=ROOT / "paper/results/retention_selection_60s_full_grid/tables/query_decomposition_common_source.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/tikz")
    parser.add_argument("--middle-panel", choices=("mean", "change"), default="mean",
                        help="Complete figure: absolute mean distances or late-minus-early changes")
    args = parser.parse_args()
    export(args.legacy, args.queries, args.output, args.middle_panel)


if __name__ == "__main__":
    main()
