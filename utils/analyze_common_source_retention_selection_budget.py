"""Measure retention/selection tradeoffs across memory budgets.

Each policy contributes its real reconstructed bank and selected frame index,
while candidate content is read from one shared DINO feature source (normally
the unbounded rollout). Ground-truth DINO features define the target. This
common-source protocol isolates candidate-set composition and retrieval from
policy-specific autoregressive histories.

The script intentionally consumes the feature cache produced by
``analyze_retrieval_quality_decomposition.py``. It does not load DINO, decode
videos, generate videos, or require a GPU.
"""

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import analyze_retrieval_quality_decomposition as decomposition  # noqa: E402


DEFAULT_RUNS = (
    "baseline",
    "fifo_b16",
    "fifo_b32",
    "fifo_b64",
    "fifo_b128",
    "ri_b16_dino_rgb",
    "ri_b32_dino_rgb",
    "ri_b64_dino_rgb",
    "ri_b128_dino_rgb",
    "slam_b16_covisibility",
    "slam_b32_covisibility",
    "slam_b64_covisibility",
    "slam_b128_covisibility",
    "kcenter_b16",
    "kcenter_b32",
    "kcenter_b64",
    "kcenter_b128",
)

FAMILY_COLORS = {
    "Unbounded": "#4A4A4A",
    "FIFO": "#C44E52",
    "RI": "#D18419",
    "GeoCov": "#23864D",
    "K-center": "#6F5AA8",
}

FAMILY_ORDER = ("FIFO", "RI", "K-center", "GeoCov")


def parse_list(value):
    return [part.strip() for part in str(value).split(",") if part.strip()]


def describe_run(run_name):
    if run_name == "baseline":
        return "Unbounded", None
    match = re.search(r"_b(\d+)(?:_|$)", run_name)
    budget = int(match.group(1)) if match else None
    if run_name.startswith("fifo_"):
        return "FIFO", budget
    if run_name.startswith("ri_"):
        return "RI", budget
    if run_name.startswith("slam_"):
        return "GeoCov", budget
    if run_name.startswith("kcenter_"):
        return "K-center", budget
    return run_name.replace("_", " "), budget


def load_cached_features(path, expected_frames):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached DINO features: {path}")
    features = np.load(path, mmap_mode="r")
    if features.ndim != 2 or len(features) < int(expected_frames):
        raise RuntimeError(
            f"Invalid feature cache {path}: shape={features.shape}, "
            f"expected at least {expected_frames} rows"
        )
    return np.asarray(features[: int(expected_frames)], dtype=np.float32)


def cache_paths(feature_cache_dir, content_run, item):
    stem = f"{item['output_prefix']}dino.npy"
    return (
        Path(feature_cache_dir) / "gt" / stem,
        Path(feature_cache_dir) / content_run / stem,
    )


def audit_inputs(items, root, runs, feature_cache_dir, content_run):
    missing = []
    for item in items:
        gt_cache, content_cache = cache_paths(feature_cache_dir, content_run, item)
        for path in (gt_cache, content_cache):
            if not path.is_file():
                missing.append(path)
        trace_name = f"{item['output_prefix']}custom.jsonl"
        for run_name in runs:
            path = Path(root) / run_name / "access_traces" / trace_name
            if not path.is_file():
                missing.append(path)
    return missing


def split_complete_items(items, root, runs, feature_cache_dir, content_run):
    complete = []
    excluded = []
    for item in items:
        missing = audit_inputs(
            [item],
            root=root,
            runs=runs,
            feature_cache_dir=feature_cache_dir,
            content_run=content_run,
        )
        if missing:
            excluded.append(
                {
                    "row": int(item["_row"]),
                    "scene": item["scene"],
                    "missing_count": len(missing),
                    "missing_paths": " | ".join(str(path) for path in missing),
                }
            )
        else:
            complete.append(item)
    return complete, excluded


def sampled_shared_queries(selected_by_run, reference_run, target_stride):
    shared = None
    for selected in selected_by_run.values():
        keys = set(selected)
        shared = keys if shared is None else shared & keys
    if not shared:
        return []
    reference = selected_by_run[reference_run]
    output = []
    for key in sorted(shared):
        section_idx, target_frame = key
        row = reference[key]
        context_slot = int(
            row.get("context_slot", target_frame - section_idx * decomposition.SECTION_STRIDE - 1)
        )
        if context_slot % int(target_stride) == 0:
            output.append((section_idx, target_frame, context_slot))
    return output


def common_source_item_rows(
    item,
    events_by_run,
    generated_features,
    gt_features,
    runs,
    reference_run,
    content_run,
    target_stride,
    strict=True,
):
    selected_by_run = {
        run_name: decomposition.selected_context_rows(events_by_run[run_name])
        for run_name in runs
    }
    if any(not rows for rows in selected_by_run.values()):
        empty = [run for run, rows in selected_by_run.items() if not rows]
        raise RuntimeError(f"No selected context rows for: {','.join(empty)}")

    sampled = sampled_shared_queries(
        selected_by_run,
        reference_run=reference_run,
        target_stride=target_stride,
    )
    if not sampled:
        raise RuntimeError(f"No shared sampled queries for row {item['_row']}")

    feature_count = min(len(generated_features), len(gt_features))
    target_frames = np.asarray([target for _section, target, _slot in sampled])
    if int(target_frames.max()) >= feature_count:
        raise IndexError(
            f"Target frame {int(target_frames.max())} exceeds {feature_count} cached features"
        )
    effective_similarities = (
        np.asarray(gt_features[target_frames], dtype=np.float32)
        @ np.asarray(generated_features, dtype=np.float32).T
    )

    banks_by_run = {}
    for run_name in runs:
        max_section = max(section for section, _target in selected_by_run[run_name])
        banks_by_run[run_name] = decomposition.reconstruct_candidate_banks(
            events_by_run[run_name],
            max_section=max_section,
            num_frames=int(item["num_frames"]),
        )

    output = []
    for run_name in runs:
        family, budget = describe_run(run_name)
        selected = selected_by_run[run_name]
        banks = banks_by_run[run_name]
        for query_index, (section_idx, target_frame, context_slot) in enumerate(sampled):
            trace_row = selected[(section_idx, target_frame)]
            bank_candidates = banks.get(section_idx, [])
            history_candidates = range(
                0,
                max(0, int(section_idx) * decomposition.SECTION_STRIDE - 3),
            )
            result = decomposition.compute_query_decomposition(
                generated_features=generated_features,
                gt_features=gt_features,
                bank_candidates=bank_candidates,
                history_candidates=history_candidates,
                selected_frame=int(trace_row["selected_memory_frame"]),
                target_frame=int(target_frame),
                effective_similarities=effective_similarities[query_index],
            )
            traced_count = int(trace_row.get("candidate_count", -1))
            count_mismatch = int(
                traced_count >= 0 and traced_count != len(bank_candidates)
            )
            if strict and count_mismatch:
                raise RuntimeError(
                    f"Bank reconstruction mismatch for row {item['_row']} "
                    f"{run_name} section {section_idx}: reconstructed="
                    f"{len(bank_candidates)}, trace={traced_count}"
                )
            output.append(
                {
                    "run_name": run_name,
                    "family": family,
                    "budget": budget,
                    "content_run": content_run,
                    "row": int(item["_row"]),
                    "scene": item["scene"],
                    "dataset_start_frame": int(item["start_frame"]),
                    "duration_sec": int(item["duration_sec"]),
                    "section_idx": int(section_idx),
                    "section_time_sec": float(
                        int(section_idx)
                        * decomposition.SECTION_STRIDE
                        / float(item["fps"])
                    ),
                    "context_slot": int(context_slot),
                    "target_frame": int(target_frame),
                    "candidate_count": len(bank_candidates),
                    "traced_candidate_count": traced_count,
                    "candidate_count_mismatch": count_mismatch,
                    "selected_overlap": trace_row.get("selected_overlap"),
                    **result,
                }
            )
    return output


def mean(values):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else None


def bootstrap_mean_interval(values, seed=17, samples=5000):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(values, size=(int(samples), len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_runs(query_rows, min_section=1):
    filtered = [row for row in query_rows if int(row["section_idx"]) >= int(min_section)]
    by_trajectory = defaultdict(list)
    for row in filtered:
        by_trajectory[(row["run_name"], int(row["row"]))].append(row)

    trajectory_rows = []
    for (run_name, row_idx), rows in sorted(by_trajectory.items()):
        family, budget = describe_run(run_name)
        trajectory_rows.append(
            {
                "run_name": run_name,
                "family": family,
                "budget": budget,
                "row": row_idx,
                "scene": rows[0]["scene"],
                "queries": len(rows),
                "retention_gap": mean(row["retention_gap"] for row in rows),
                "retrieval_gap": mean(row["retrieval_gap"] for row in rows),
                "total_oracle_gap": mean(row["total_oracle_gap"] for row in rows),
                "candidate_count": mean(row["candidate_count"] for row in rows),
            }
        )

    by_run = defaultdict(list)
    for row in trajectory_rows:
        by_run[row["run_name"]].append(row)
    summaries = []
    for run_name, rows in sorted(by_run.items()):
        family, budget = describe_run(run_name)
        summary = {
            "run_name": run_name,
            "family": family,
            "budget": budget,
            "min_section": int(min_section),
            "trajectories": len(rows),
            "queries": sum(int(row["queries"]) for row in rows),
            "candidate_count": mean(row["candidate_count"] for row in rows),
        }
        for metric in ("retention_gap", "retrieval_gap", "total_oracle_gap"):
            values = [float(row[metric]) for row in rows]
            ci_low, ci_high = bootstrap_mean_interval(values)
            summary[metric] = mean(values)
            summary[f"{metric}_ci_low"] = ci_low
            summary[f"{metric}_ci_high"] = ci_high
        summaries.append(summary)
    return summaries, trajectory_rows


def family_budget_steps(summary_rows):
    output = []
    grouped = defaultdict(list)
    for row in summary_rows:
        if row["budget"] is not None:
            grouped[row["family"]].append(row)
    for family, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["budget"]))
        for left, right in zip(rows, rows[1:]):
            output.append(
                {
                    "family": family,
                    "from_budget": int(left["budget"]),
                    "to_budget": int(right["budget"]),
                    "retention_gap_change": float(right["retention_gap"])
                    - float(left["retention_gap"]),
                    "retrieval_gap_change": float(right["retrieval_gap"])
                    - float(left["retrieval_gap"]),
                    "moves_left": float(right["retention_gap"])
                    < float(left["retention_gap"]),
                    "moves_down": float(right["retrieval_gap"])
                    < float(left["retrieval_gap"]),
                }
            )
    return output


def plot_tradeoff(summary_rows, output_path, title):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_budget_tradeoff_mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/memcam_budget_tradeoff_xdg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row["family"]].append(row)

    for family in FAMILY_ORDER:
        rows = sorted(
            [row for row in grouped.get(family, []) if row["budget"] is not None],
            key=lambda row: int(row["budget"]),
        )
        if not rows:
            continue
        xs = [float(row["retention_gap"]) for row in rows]
        ys = [float(row["retrieval_gap"]) for row in rows]
        color = FAMILY_COLORS[family]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.0,
            alpha=0.75,
            zorder=2,
            label=family,
        )
        for position, row in enumerate(rows):
            x = float(row["retention_gap"])
            y = float(row["retrieval_gap"])
            xerr = np.asarray(
                [
                    [x - float(row["retention_gap_ci_low"])],
                    [float(row["retention_gap_ci_high"]) - x],
                ]
            )
            yerr = np.asarray(
                [
                    [y - float(row["retrieval_gap_ci_low"])],
                    [float(row["retrieval_gap_ci_high"]) - y],
                ]
            )
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt="o",
                markersize=9,
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.5,
                alpha=0.92,
                zorder=3,
            )
            offset_y = 8 if position % 2 == 0 else -15
            ax.annotate(
                f"B{int(row['budget'])}",
                (x, y),
                xytext=(7, offset_y),
                textcoords="offset points",
                color=color,
                fontsize=9,
                weight="bold",
            )
        for left, right in zip(rows, rows[1:]):
            ax.annotate(
                "",
                xy=(float(right["retention_gap"]), float(right["retrieval_gap"])),
                xytext=(float(left["retention_gap"]), float(left["retrieval_gap"])),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "linewidth": 1.6,
                    "shrinkA": 7,
                    "shrinkB": 7,
                },
                zorder=4,
            )

    for row in grouped.get("Unbounded", []):
        x = float(row["retention_gap"])
        y = float(row["retrieval_gap"])
        ax.scatter(
            x,
            y,
            marker="*",
            s=240,
            color=FAMILY_COLORS["Unbounded"],
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
            label="Unbounded",
        )
        ax.annotate(
            "Unbounded",
            (x, y),
            xytext=(8, -3),
            textcoords="offset points",
            color=FAMILY_COLORS["Unbounded"],
            fontsize=9.5,
            weight="bold",
        )

    ax.annotate(
        "ideal direction",
        xy=(0.02, 0.04),
        xycoords="axes fraction",
        xytext=(0.20, 0.18),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#707070", "linewidth": 1.2},
        fontsize=9,
        color="#555555",
    )
    ax.set_xlabel("Retention gap: useful history deleted (lower is better)")
    ax.set_ylabel("Selection gap: retained evidence unused (lower is better)")
    ax.set_title(title, loc="left", fontsize=13, weight="bold")
    ax.text(
        0.0,
        1.01,
        "Arrows follow increasing budget B16 -> B32 -> B64 -> B128; bars are trajectory-bootstrap 95% CIs",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
        va="bottom",
    )
    ax.grid(color="#DADCE0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="best")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_summary_table(rows):
    lines = [
        "| policy | budget | trajectories | candidates | retention gap | 95% CI | selection gap | 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    order = {
        "Unbounded": 0,
        "FIFO": 1,
        "RI": 2,
        "K-center": 3,
        "GeoCov": 4,
    }
    for row in sorted(
        rows,
        key=lambda item: (
            order.get(item["family"], 9),
            item["budget"] if item["budget"] is not None else 0,
        ),
    ):
        budget = "--" if row["budget"] is None else str(int(row["budget"]))
        lines.append(
            f"| {row['family']} | {budget} | {row['trajectories']} | "
            f"{row['candidate_count']:.1f} | {row['retention_gap']:.4f} | "
            f"[{row['retention_gap_ci_low']:.4f}, {row['retention_gap_ci_high']:.4f}] | "
            f"{row['retrieval_gap']:.4f} | "
            f"[{row['retrieval_gap_ci_low']:.4f}, {row['retrieval_gap_ci_high']:.4f}] |"
        )
    return lines


def write_report(
    path,
    overall,
    late,
    steps,
    content_run,
    target_stride,
    late_section,
    matched_rows,
    excluded_rows,
):
    lines = [
        "# Common-Source Retention--Selection Budget Sweep",
        "",
        "## Protocol",
        "",
        f"Every policy supplies its real reconstructed bank and selected indices. Candidate image features for every policy come from the same `{content_run}` rollout. Target features come from exact-index dataset ground truth. One of every `{target_stride}` context slots is sampled.",
        "",
        "This isolates candidate-set retention and selection from policy-specific generated histories. The DINO hindsight-best candidate is a diagnostic proxy, not a deployable oracle.",
        "",
        f"Matched manifest rows used by every plotted policy: `{','.join(str(row) for row in matched_rows)}` (`n={len(matched_rows)}`).",
        f"Rows excluded because at least one requested trace/cache was missing: `{','.join(str(row['row']) for row in excluded_rows) if excluded_rows else 'none'}`.",
        "",
        "## Whole Rollout",
        "",
        *format_summary_table(overall),
        "",
        f"## Late Rollout (section >= {late_section})",
        "",
        *format_summary_table(late),
        "",
        "## Increasing-Budget Steps",
        "",
        "Negative changes are improvements. `moves_left` means the larger budget reduced retention loss; `moves_down` means it also reduced selection loss.",
        "",
        "| family | budget step | retention change | selection change | moves left | moves down |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in steps:
        lines.append(
            f"| {row['family']} | B{row['from_budget']} -> B{row['to_budget']} | "
            f"{row['retention_gap_change']:+.4f} | {row['retrieval_gap_change']:+.4f} | "
            f"{row['moves_left']} | {row['moves_down']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `figures/retention_selection_budget_all.png`",
            "- `figures/retention_selection_budget_late.png`",
            "- `tables/run_summary_all.csv`",
            "- `tables/run_summary_late.csv`",
            "- `tables/trajectory_summary_all.csv`",
            "- `tables/query_decomposition_common_source.csv`",
            "- `tables/increasing_budget_steps.csv`",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    parser.add_argument("--reference_run", default="baseline")
    parser.add_argument("--content_run", default="baseline")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--rows", default=None)
    parser.add_argument("--target_stride", type=int, default=19)
    parser.add_argument("--late_section", type=int, default=35)
    parser.add_argument("--feature_cache_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--allow_incomplete_rows",
        action="store_true",
        help=(
            "Use the intersection of trajectories with every requested trace/cache. "
            "The report records all excluded rows."
        ),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    runs = parse_list(args.runs)
    if args.reference_run not in runs:
        raise ValueError("--reference_run must be included in --runs")
    if args.target_stride <= 0:
        raise ValueError("--target_stride must be positive")
    items = decomposition.load_manifest(
        args.manifest,
        duration=args.duration,
        selected_rows=decomposition.parse_rows(args.rows),
    )
    if not items:
        raise RuntimeError("No manifest rows selected")

    complete_items, excluded_rows = split_complete_items(
        items,
        root=args.root,
        runs=runs,
        feature_cache_dir=args.feature_cache_dir,
        content_run=args.content_run,
    )
    if excluded_rows and not args.allow_incomplete_rows:
        missing = [
            path
            for row in excluded_rows
            for path in row["missing_paths"].split(" | ")
        ]
        preview = "\n".join(missing[:30])
        raise FileNotFoundError(
            f"Missing {len(missing)} traces or cached feature files. "
            f"The CPU sweep does not silently re-encode DINO:\n{preview}"
        )
    if excluded_rows:
        print("Using the common complete trajectory subset:")
        for row in excluded_rows:
            print(
                f"  excluded row {row['row']} {row['scene']}: "
                f"{row['missing_count']} missing input(s)"
            )
        items = complete_items
    if not items:
        raise RuntimeError("No trajectory has every requested trace and feature cache")
    print(f"Matched rows used for every policy: {[int(item['_row']) for item in items]}")

    query_rows = []
    for position, item in enumerate(items, start=1):
        expected_frames = int(item["num_frames"])
        gt_cache, content_cache = cache_paths(
            args.feature_cache_dir,
            args.content_run,
            item,
        )
        print(
            f"[{position}/{len(items)}] row {item['_row']} {item['scene']}: "
            "loading shared caches"
        )
        gt_features = load_cached_features(gt_cache, expected_frames)
        generated_features = load_cached_features(content_cache, expected_frames)
        events_by_run = {}
        trace_name = f"{item['output_prefix']}custom.jsonl"
        for run_name in runs:
            trace_path = args.root / run_name / "access_traces" / trace_name
            events_by_run[run_name] = decomposition.read_trace(
                trace_path,
                expected_identity=decomposition.run_identity(item),
            )
        rows = common_source_item_rows(
            item=item,
            events_by_run=events_by_run,
            generated_features=generated_features,
            gt_features=gt_features,
            runs=runs,
            reference_run=args.reference_run,
            content_run=args.content_run,
            target_stride=args.target_stride,
            strict=args.strict,
        )
        query_rows.extend(rows)
        print(f"  wrote {len(rows)} run-query rows")

    overall, trajectories = summarize_runs(query_rows, min_section=1)
    late, late_trajectories = summarize_runs(
        query_rows,
        min_section=args.late_section,
    )
    steps = family_budget_steps(overall)

    tables = args.output_dir / "tables"
    figures = args.output_dir / "figures"
    write_csv(tables / "query_decomposition_common_source.csv", query_rows)
    write_csv(tables / "trajectory_summary_all.csv", trajectories)
    write_csv(tables / "trajectory_summary_late.csv", late_trajectories)
    write_csv(tables / "run_summary_all.csv", overall)
    write_csv(tables / "run_summary_late.csv", late)
    write_csv(tables / "increasing_budget_steps.csv", steps)
    plot_tradeoff(
        overall,
        figures / "retention_selection_budget_all.png",
        "Retention--selection tradeoff across memory budgets",
    )
    plot_tradeoff(
        late,
        figures / "retention_selection_budget_late.png",
        "Late-horizon retention--selection tradeoff",
    )
    write_report(
        args.output_dir / "report.md",
        overall,
        late,
        steps,
        content_run=args.content_run,
        target_stride=args.target_stride,
        late_section=args.late_section,
        matched_rows=[int(item["_row"]) for item in items],
        excluded_rows=excluded_rows,
    )
    print(f"Wrote: {args.output_dir / 'report.md'}")
    print(f"Wrote: {figures / 'retention_selection_budget_all.png'}")
    print(f"Wrote: {figures / 'retention_selection_budget_late.png'}")


if __name__ == "__main__":
    main()
