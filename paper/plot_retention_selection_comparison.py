"""Redraw the original tradeoff with explicitly labeled, mixed rollout horizons."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


ROOT = Path(__file__).resolve().parent
BUDGETS = (16, 32, 64, 128)
POLICIES = ("Unbounded", "FIFO", "K-center", "MCE", "KEEPSAKE")
COLORS = dict(zip(POLICIES, ("#454545", "#C56368", "#BF902E", "#8C79AF", "#27845D")))
RUNS = {"baseline": ("Unbounded", None)}
for policy, template in (("FIFO", "fifo_b{}"), ("K-center", "kcenter_b{}"),
                         ("MCE", "mce_b{}_lambda1_pilot")):
    RUNS.update({template.format(b): (policy, b) for b in BUDGETS})
METRICS = ("retention_gap", "retrieval_gap", "total_oracle_gap")


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_gaps(values):
    if not all(math.isfinite(v) and v >= -1e-6 for v in values):
        raise ValueError(f"Invalid gap values: {values}")
    if not math.isclose(values[0] + values[1], values[2], abs_tol=1e-6):
        raise ValueError(f"Retention and selection do not sum to total: {values}")


def load_points(summary_path, queries_path, legacy_path):
    summaries = {}
    for row in read_csv(summary_path):
        run = row["run_name"]
        if run not in RUNS:
            continue
        if run in summaries:
            raise ValueError(f"Duplicate summary: {run}")
        if int(row["trajectories"]) != 15 or int(row["min_section"]) != 1:
            raise ValueError(f"Expected whole-rollout, fifteen-trajectory summary: {run}")
        values = tuple(float(row[m]) for m in METRICS)
        validate_gaps(values)
        expected_budget = RUNS[run][1]
        if (int(row["budget"]) if row["budget"] else None) != expected_budget:
            raise ValueError(f"Wrong recorded budget: {run}")
        summaries[run] = row
    if set(summaries) != set(RUNS):
        raise ValueError(f"Missing 60-second summaries: {sorted(set(RUNS) - set(summaries))}")

    # Check the recorded horizon, common queries, and equal-trajectory means.
    queries = {run: {} for run in RUNS}
    for row in read_csv(queries_path):
        run = row["run_name"]
        if run not in RUNS:
            continue
        if int(row["duration_sec"]) != 60 or row["content_run"] != "baseline":
            raise ValueError(f"Expected common-baseline 60-second queries: {run}")
        if int(row["candidate_count_mismatch"]) != 0:
            raise ValueError(f"Candidate count mismatch: {run}")
        identity = (row["row"], row["scene"], row["dataset_start_frame"])
        key = identity + (row["section_idx"], row["target_frame"])
        if key in queries[run]:
            raise ValueError(f"Duplicate query: {run}, {key}")
        values = tuple(float(row[m]) for m in METRICS)
        validate_gaps(values)
        queries[run][key] = values

    points = []
    for run, (policy, budget) in RUNS.items():
        if queries[run].keys() != queries["baseline"].keys():
            raise ValueError(f"Mismatched query coverage: {run}")
        by_trajectory = defaultdict(list)
        for key, values in queries[run].items():
            by_trajectory[key[:3]].append(values)
        summary = summaries[run]
        if len(by_trajectory) != 15 or len(queries[run]) != int(summary["queries"]):
            raise ValueError(f"Wrong cohort or query count: {run}")
        for column, metric in enumerate(METRICS):
            mean = sum(sum(v[column] for v in group) / len(group)
                       for group in by_trajectory.values()) / len(by_trajectory)
            if not math.isclose(mean, float(summary[metric]), abs_tol=1e-8):
                raise ValueError(f"Summary does not match query means: {run}, {metric}")
        if policy == "Unbounded" and abs(float(summary["retention_gap"])) > 1e-6:
            raise ValueError("Unbounded must have zero retention gap")
        points.append(dict(policy=policy, run=run, budget=budget, duration_sec=60,
                           trajectories=15, queries=len(queries[run]),
                           retention_gap=float(summary["retention_gap"]),
                           selection_gap=float(summary["retrieval_gap"]),
                           total_gap=float(summary["total_oracle_gap"]),
                           source_kind="validated_query_summary"))

    legacy = json.loads(legacy_path.read_text())
    if (legacy["policy"], legacy["duration_sec"], legacy["trajectories"],
        legacy["source_kind"]) != ("KEEPSAKE", 180, 13, "legacy_rounded_summary"):
        raise ValueError("Expected original KEEPSAKE 180-second, thirteen-trajectory points")
    if sorted(p["budget"] for p in legacy["points"]) != list(BUDGETS):
        raise ValueError("Expected exactly one legacy point at each budget")
    for point in sorted(legacy["points"], key=lambda p: p["budget"]):
        retention, selection = float(point["retention_gap"]), float(point["selection_gap"])
        validate_gaps((retention, selection, retention + selection))
        points.append(dict(policy="KEEPSAKE", run=f"slam_b{point['budget']}_covisibility",
                           budget=point["budget"], duration_sec=180, trajectories=13,
                           queries=None, retention_gap=retention, selection_gap=selection,
                           total_gap=retention + selection, source_kind=legacy["source_kind"]))
    return points


def make_figure(points):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.25))
    fig.subplots_adjust(left=.075, right=.985, bottom=.15, top=.78, wspace=.20)
    offsets = {
        ("K-center", 16): (-32, -15), ("K-center", 32): (7, 15),
        ("K-center", 64): (-30, -14), ("K-center", 128): (6, 8),
        ("MCE", 16): (6, 6), ("MCE", 32): (28, 6),
        ("MCE", 64): (-20, -25), ("MCE", 128): (-18, 20),
        ("KEEPSAKE", 16): (6, -3), ("KEEPSAKE", 32): (8, -5),
        ("KEEPSAKE", 64): (-44, -6), ("KEEPSAKE", 128): (7, 10),
    }
    for index, ax in enumerate(axes):
        for policy in POLICIES:
            if index == 1 and policy in ("Unbounded", "FIFO"):
                continue
            rows = [p for p in points if p["policy"] == policy]
            x, y = ([p[m] for p in rows] for m in ("retention_gap", "selection_gap"))
            color = COLORS[policy]
            ax.plot(x, y, color=color, lw=1.5, marker="o", markersize=7,
                    markeredgecolor="white", markeredgewidth=.9,
                    linestyle="--" if policy == "KEEPSAKE" else "-", zorder=3)
            for p in rows:
                if policy == "Unbounded":
                    label, offset = "Unbounded", (9, -3)
                elif index == 0 and policy != "FIFO":
                    continue
                else:
                    label = f"B{p['budget']}"
                    offset = offsets.get((policy, p["budget"]), (7, -3))
                leader = (dict(arrowstyle="-", color=color, lw=.55, shrinkA=3, shrinkB=6)
                          if index == 1 and max(abs(v) for v in offset) >= 14 else None)
                ax.annotate(label, (p["retention_gap"], p["selection_gap"]),
                            xytext=offset, textcoords="offset points", color=color,
                            fontsize=9, weight="medium", zorder=4, arrowprops=leader,
                            bbox=dict(facecolor="white", edgecolor="none", pad=.3, alpha=.9))
        ax.set_xlabel("Retention gap (lower is better)", labelpad=9)
        ax.grid(color="#E5E7EB", linewidth=.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_locator(MultipleLocator(.04 if index == 0 else .02))
    axes[0].set(xlim=(-.008, .202), ylim=(.025, .216),
                ylabel="Selection gap (lower is better)")
    axes[0].xaxis.set_major_locator(MultipleLocator(.04))
    axes[1].set(xlim=(.008, .076), ylim=(.093, .201))
    axes[1].xaxis.set_major_locator(MultipleLocator(.02))
    axes[0].set_title("(a) Full policy landscape", loc="left", pad=12)
    axes[1].set_title("(b) K-center, MCE and KEEPSAKE", loc="left", pad=12)
    handles = []
    for policy in POLICIES:
        row = next(p for p in points if p["policy"] == policy)
        handles.append(Line2D([], [], color=COLORS[policy], lw=1.5, marker="o",
                              linestyle="--" if policy == "KEEPSAKE" else "-",
                              label=f"{policy}\n{row['duration_sec']} s, n={row['trajectories']}"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .995),
               ncol=5, frameon=False, handlelength=2.2, columnspacing=2.1, fontsize=10)
    return fig


def export(summary, queries, legacy, output):
    points = load_points(summary, queries, legacy)
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.labelsize": 11, "axes.linewidth": .7,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig = make_figure(points)
        for suffix in ("png", "pdf"):
            fig.savefig(output.with_suffix("." + suffix), dpi=240, facecolor="white")
        plt.close(fig)
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    caption = (
        "Retention-selection tradeoffs across budgets B=16, 32, 64 and 128. "
        "Unbounded, FIFO, K-center and MCE: 60 seconds, 15 matched trajectories; "
        "KEEPSAKE (dashed): 180 seconds, 13 trajectories. "
        "Points are within-suite common-source means; lines connect budgets. "
        "The right panel enlarges K-center, MCE and KEEPSAKE. "
        "Different horizons and cohorts make this a descriptive, not matched-duration, comparison."
    )
    output.with_suffix(".caption.txt").write_text(caption + "\n")
    output.with_suffix(".provenance.json").write_text(json.dumps({
        "system": "MemCam", "comparison": "mixed_horizon_descriptive",
        "policies": POLICIES, "budgets": BUDGETS, "points": points,
        "inputs": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in (summary, queries, legacy)],
        "legacy_source": json.loads(legacy.read_text()),
        "definitions": {"retention_gap": "bank oracle minus full-history oracle",
                        "selection_gap": "selected distance minus bank oracle",
                        "total_gap": "retention_gap + selection_gap"},
        "limits": ["Not a matched-duration or matched-cohort policy comparison.",
                   "180s values reproduce rounded legacy points, not a new computation.",
                   "No uncertainty is plotted; no intervals inferred for legacy aggregates.",
                   "60s query identities, counts and means checked; raw videos not re-audited."],
    }, indent=2) + "\n")
    print(f"Wrote {output.with_suffix('.pdf')} and .png ({len(points)} points)")


def main():
    tables = ROOT / "retention_selection_60s_full_grid/tables"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=tables / "run_summary_all.csv")
    parser.add_argument("--queries", type=Path, default=tables / "query_decomposition_common_source.csv")
    parser.add_argument("--legacy", type=Path, default=ROOT / "retention_selection_180s_reported.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "figures/retention_selection_comparison_mixed_horizons")
    args = parser.parse_args()
    export(args.summary, args.queries, args.legacy, args.output)


if __name__ == "__main__":
    main()
