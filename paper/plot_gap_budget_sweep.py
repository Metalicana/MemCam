#!/usr/bin/env python3
"""Restyle saved gap summaries without recomputing any estimates or intervals."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


COLORS = {
    "Unbounded": "#454545",
    "FIFO": "#BF555A",
    "RI": "#BA8524",
    "K-center": "#8171AA",
    "MCE": "#37829E",
    "Keepsake": "#238356",
}
MARKERS = {16: "o", 32: "s", 64: "^", 128: "D"}


def read_summary(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    for row in rows:
        family = row["family"]
        if family == "GeoCov":
            family = "Keepsake"
        if row["run_name"].startswith("mce_"):
            family = "MCE"
        if family not in COLORS:
            raise ValueError(f"Unsupported family: {family}")
        row["display_family"] = family
        row["budget"] = int(row["budget"]) if row["budget"] else None
        key = family, row["budget"]
        if key in seen:
            raise ValueError(f"Duplicate policy/budget: {key}")
        seen.add(key)
        if family != "Unbounded" and row["budget"] not in MARKERS:
            raise ValueError(f"Unsupported budget: {key}")
        for metric in ("retention_gap", "retrieval_gap"):
            for suffix in ("", "_ci_low", "_ci_high"):
                name = metric + suffix
                row[name] = float(row[name])
                if not math.isfinite(row[name]):
                    raise ValueError(f"Nonfinite {name}: {key}")
            if not row[metric + "_ci_low"] <= row[metric] <= row[metric + "_ci_high"]:
                raise ValueError(f"Interval does not enclose estimate: {key}, {metric}")
    if not rows:
        raise ValueError("Empty summary")
    return rows


def plot_summary(source, output, title):
    rows = read_summary(source)
    with plt.rc_context({"font.size": 9, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(7.4, 5.4))
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.23, top=0.79)
        handles = []
        for family, color in COLORS.items():
            group = sorted((r for r in rows if r["display_family"] == family),
                           key=lambda r: r["budget"] or 0)
            if not group:
                continue
            xs = [r["retention_gap"] for r in group]
            ys = [r["retrieval_gap"] for r in group]
            if family != "Unbounded":
                ax.plot(xs, ys, color=color, linewidth=0.75, alpha=0.8, zorder=3)
            for row, x, y in zip(group, xs, ys):
                # Preserve full marginal intervals; only their visual weight changes.
                ax.errorbar(x, y,
                            xerr=[[x - row["retention_gap_ci_low"]],
                                  [row["retention_gap_ci_high"] - x]],
                            yerr=[[y - row["retrieval_gap_ci_low"]],
                                  [row["retrieval_gap_ci_high"] - y]],
                            fmt="none", ecolor=color, elinewidth=0.55,
                            alpha=0.23, capsize=0, zorder=1)
                marker = "*" if family == "Unbounded" else MARKERS[row["budget"]]
                ax.plot(x, y, marker=marker, linestyle="none", color=color,
                        markersize=8 if family == "Unbounded" else 5,
                        markeredgecolor="white", markeredgewidth=0.45, zorder=4)
            handles.append(Line2D([], [], color=color, linewidth=0.9,
                                  marker="*" if family == "Unbounded" else "o",
                                  markersize=5, label=family))
        fig.suptitle(title, x=0.12, y=0.97, ha="left", fontsize=12)
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.11, 0.925),
                   ncol=6, frameon=False, handlelength=1.25, columnspacing=1.2)
        budget_handles = [Line2D([], [], color="#555555", marker=marker,
                                 linestyle="none", markersize=5, label=f"B{budget}")
                          for budget, marker in MARKERS.items()]
        fig.legend(handles=budget_handles, loc="lower center", bbox_to_anchor=(0.55, 0.075),
                   ncol=4, frameon=False, handletextpad=0.3, columnspacing=1.8)
        ax.set_xlabel("Retention gap (lower is better)", labelpad=8)
        ax.set_ylabel("Selection gap (lower is better)", labelpad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(width=0.6, length=3, labelsize=8)
        ax.grid(color="#EEEEEE", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.margins(x=0.04, y=0.07)
        fig.text(0.12, 0.035, "Faint bars: trajectory-bootstrap 95% CIs. Lines connect budgets; lower-left is better.",
                 fontsize=8, color="#666666")
        output.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            fig.savefig(output.with_suffix("." + extension), dpi=240, facecolor="white")
        plt.close(fig)
    print(f"Rendered {len(rows)} configurations: {output.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results/retention_selection_60s_full_grid")
    args = parser.parse_args()
    for period, title in (("all", "Retention and selection across memory budgets"),
                          ("late", "Retention and selection across memory budgets: late rollout")):
        plot_summary(args.results / "tables" / f"run_summary_{period}.csv",
                     args.results / "figures" / f"retention_selection_budget_{period}", title)


if __name__ == "__main__":
    main()
