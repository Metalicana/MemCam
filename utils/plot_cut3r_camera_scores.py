import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "memcam_matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "memcam_xdg_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    ("rotation_error_deg_mean_mean", "Rotation Error Mean", "degrees", "lower"),
    ("translation_error_scale_only_mean_mean", "Translation Error Mean", "trajectory units", "lower"),
    ("translation_error_sim3_mean_mean", "Sim3 Translation Error Mean", "trajectory units", "lower"),
    ("endpoint_rotation_error_deg_mean", "Endpoint Rotation Error", "degrees", "lower"),
    ("loop_endpoint_distance_error_mean", "Loop Endpoint Distance Error", "trajectory units", "lower"),
    ("worldscore_camera_control_score_mean", "WorldScore-Style Camera Score", "score", "higher"),
]

DEFAULT_LABELS = {
    "baseline": "Unbounded",
    "fifo_b32": "FIFO",
    "fifo_b64": "FIFO",
    "slam_b32_covisibility": "SLAM",
    "slam_b64_covisibility": "SLAM",
    "ri_b32_dino_rgb": "Mine",
    "ri_b64_dino_rgb": "Mine",
}

POLICY_COLORS = {
    "Unbounded": "#7A828F",
    "FIFO": "#A3BEFA",
    "SLAM": "#A3D576",
    "Mine": "#F0986E",
}

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}


def parse_list(value):
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_named_paths(value):
    output = []
    for part in parse_list(value):
        if "=" not in part:
            raise ValueError(f"Expected label=path, got: {part}")
        label, path = part.split("=", 1)
        output.append((label.strip(), Path(path).expanduser()))
    return output


def parse_run_labels(value):
    labels = dict(DEFAULT_LABELS)
    for part in parse_list(value):
        if "=" not in part and ":" not in part:
            labels[part] = DEFAULT_LABELS.get(part, part)
            continue
        if "=" in part:
            run_name, label = part.split("=", 1)
        else:
            run_name, label = part.split(":", 1)
        labels[run_name.strip()] = label.strip()
    return labels


def parse_metric_specs(value):
    if not value:
        return METRICS
    metric_names = parse_list(value)
    known = {name: (name, title, unit, direction) for name, title, unit, direction in METRICS}
    specs = []
    for name in metric_names:
        specs.append(known.get(name, (name, name, "", "lower")))
    return specs


def safe_float(value):
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def read_summary(label, path, run_labels, run_filter):
    if not path.exists():
        raise FileNotFoundError(f"CUT3R summary not found: {path}")
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            run_name = row.get("run_name")
            if run_filter and run_name not in run_filter:
                continue
            rows.append(
                {
                    **row,
                    "budget": label,
                    "label": run_labels.get(run_name, run_name),
                    "source_path": str(path),
                }
            )
    return rows


def long_rows(summary_rows, metric_specs):
    output = []
    for row in summary_rows:
        for metric, title, unit, direction in metric_specs:
            value = safe_float(row.get(metric))
            output.append(
                {
                    "budget": row["budget"],
                    "run_name": row["run_name"],
                    "label": row["label"],
                    "metric": metric,
                    "metric_title": title,
                    "unit": unit,
                    "direction": direction,
                    "value": value,
                    "videos": row.get("videos"),
                    "sampled_frames_mean": row.get("sampled_frames_mean"),
                    "source_path": row.get("source_path"),
                }
            )
    return output


def relative_rows(rows, reference_run):
    by_key = {
        (row["budget"], row["run_name"], row["metric"]): row
        for row in rows
    }
    output = []
    for row in rows:
        if row["run_name"] == reference_run:
            continue
        reference = by_key.get((row["budget"], reference_run, row["metric"]))
        reference_value = safe_float(reference.get("value")) if reference else None
        value = safe_float(row.get("value"))
        improvement = None
        delta = None
        if value is not None and reference_value is not None and abs(reference_value) > 1e-12:
            delta = value - reference_value
            if row["direction"] == "higher":
                improvement = 100.0 * (value - reference_value) / abs(reference_value)
            else:
                improvement = 100.0 * (reference_value - value) / abs(reference_value)
        output.append(
            {
                **row,
                "reference_run": reference_run,
                "reference_value": reference_value,
                "delta_value_minus_reference": delta,
                "improvement_pct_vs_reference": improvement,
            }
        )
    return output


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def setup_axis(ax):
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(axis="y", color=TOKENS["grid"], linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(colors=TOKENS["muted"])


def add_header(fig, title, subtitle):
    fig.text(0.06, 0.96, title, ha="left", va="top", fontsize=16, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.06, 0.91, subtitle, ha="left", va="top", fontsize=10.5, color=TOKENS["muted"])


def budget_order(rows):
    values = []
    for row in rows:
        if row["budget"] not in values:
            values.append(row["budget"])
    return values


def run_order(rows):
    preferred = ["Unbounded", "FIFO", "SLAM", "Mine"]
    labels = []
    for label in preferred:
        if any(row["label"] == label for row in rows):
            labels.append(label)
    for row in rows:
        if row["label"] not in labels:
            labels.append(row["label"])
    return labels


def plot_metric(rows, metric, title, unit, output_dir):
    metric_rows = [row for row in rows if row["metric"] == metric and safe_float(row.get("value")) is not None]
    if not metric_rows:
        return None

    budgets = budget_order(metric_rows)
    labels = run_order(metric_rows)
    width = 0.74 / max(1, len(labels))
    x_positions = list(range(len(budgets)))

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    fig.patch.set_facecolor(TOKENS["surface"])
    setup_axis(ax)

    for label_index, label in enumerate(labels):
        offset = (label_index - (len(labels) - 1) / 2) * width
        values = []
        for budget in budgets:
            row = next((item for item in metric_rows if item["budget"] == budget and item["label"] == label), None)
            values.append(safe_float(row.get("value")) if row else None)
        xs = [x + offset for x in x_positions]
        ax.bar(
            xs,
            [value if value is not None else 0.0 for value in values],
            width=width * 0.92,
            label=label,
            color=POLICY_COLORS.get(label, "#C5CAD3"),
            edgecolor="#464C55",
            linewidth=0.7,
        )
        for x, value in zip(xs, values):
            if value is None:
                continue
            ax.text(
                x,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=TOKENS["muted"],
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(budgets, color=TOKENS["ink"])
    ax.set_ylabel(unit or "value", color=TOKENS["ink"])
    add_header(
        fig,
        title,
        "CUT3R camera trajectory metric on 60s generated videos; compare policies within each budget.",
    )
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.08), ncol=min(len(labels), 4))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path = output_dir / f"cut3r_{metric}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote: {path}")
    return path


def plot_relative_metric(rows, metric, title, output_dir, reference_run):
    metric_rows = [
        row
        for row in rows
        if row["metric"] == metric and safe_float(row.get("improvement_pct_vs_reference")) is not None
    ]
    if not metric_rows:
        return None

    budgets = budget_order(metric_rows)
    labels = run_order(metric_rows)
    labels = [label for label in labels if label != "Unbounded"]
    width = 0.74 / max(1, len(labels))
    x_positions = list(range(len(budgets)))

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    fig.patch.set_facecolor(TOKENS["surface"])
    setup_axis(ax)
    ax.axhline(0.0, color=TOKENS["muted"], linestyle="--", linewidth=1.0)

    for label_index, label in enumerate(labels):
        offset = (label_index - (len(labels) - 1) / 2) * width
        values = []
        for budget in budgets:
            row = next((item for item in metric_rows if item["budget"] == budget and item["label"] == label), None)
            values.append(safe_float(row.get("improvement_pct_vs_reference")) if row else None)
        xs = [x + offset for x in x_positions]
        ax.bar(
            xs,
            [value if value is not None else 0.0 for value in values],
            width=width * 0.92,
            label=label,
            color=POLICY_COLORS.get(label, "#C5CAD3"),
            edgecolor="#464C55",
            linewidth=0.7,
        )
        for x, value in zip(xs, values):
            if value is None:
                continue
            va = "bottom" if value >= 0 else "top"
            ax.text(
                x,
                value,
                f"{value:+.1f}%",
                ha="center",
                va=va,
                fontsize=8,
                color=TOKENS["muted"],
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(budgets, color=TOKENS["ink"])
    ax.set_ylabel("Improvement vs Unbounded (%)", color=TOKENS["ink"])
    add_header(
        fig,
        f"{title}: improvement vs Unbounded",
        "Positive is better. Lower-error metrics are converted so reductions become positive improvements.",
    )
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.08), ncol=min(len(labels), 4))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path = output_dir / f"cut3r_{metric}_improvement_vs_{reference_run}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Plot CUT3R camera score summaries for budget comparisons."
    )
    parser.add_argument(
        "--summary_csvs",
        required=True,
        help="Comma-separated label=path entries, e.g. b32=/path/csv,b64=/path/csv.",
    )
    parser.add_argument("--runs", type=str, default=None, help="Optional comma-separated run filter.")
    parser.add_argument("--run_labels", type=str, default=None)
    parser.add_argument("--metrics", type=str, default=None)
    parser.add_argument("--reference_run", type=str, default="baseline")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_filter = set(parse_list(args.runs)) if args.runs else None
    run_labels = parse_run_labels(args.run_labels)
    metric_specs = parse_metric_specs(args.metrics)

    summary_rows = []
    for label, path in parse_named_paths(args.summary_csvs):
        summary_rows.extend(read_summary(label, path, run_labels, run_filter))
    if not summary_rows:
        raise RuntimeError("No CUT3R summary rows loaded.")

    rows = long_rows(summary_rows, metric_specs)
    rel_rows = relative_rows(rows, args.reference_run)
    write_csv(args.output_dir / "cut3r_camera_budget_scores.csv", rows)
    write_csv(args.output_dir / f"cut3r_camera_budget_scores_relative_to_{args.reference_run}.csv", rel_rows)
    print(f"Wrote: {args.output_dir / 'cut3r_camera_budget_scores.csv'}")
    print(f"Wrote: {args.output_dir / f'cut3r_camera_budget_scores_relative_to_{args.reference_run}.csv'}")

    for metric, title, unit, _direction in metric_specs:
        plot_metric(rows, metric, title, unit, args.output_dir)
        plot_relative_metric(rel_rows, metric, title, args.output_dir, args.reference_run)


if __name__ == "__main__":
    main()
