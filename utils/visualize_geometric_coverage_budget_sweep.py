"""Compare actual Geometric Coverage banks across memory budgets.

Each budget contributes retained frame identities reconstructed from its real
generation trace. All banks are evaluated against the same historical frame
universe using pixels from one common rollout, so differences in the plotted
coverage cannot come from policy-specific generated image histories.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import visualize_geometric_coverage_evictions as geometric  # noqa: E402


DEFAULT_RUNS = (
    "slam_b16_covisibility",
    "slam_b32_covisibility",
    "slam_b64_covisibility",
    "slam_b128_covisibility",
)


def parse_list(value):
    return [part.strip() for part in str(value).split(",") if part.strip()]


def run_budget(run_name):
    match = re.search(r"(?:^|_)b(\d+)(?:_|$)", str(run_name))
    if match is None:
        raise ValueError(f"Could not infer budget from run name: {run_name}")
    return int(match.group(1))


def validate_runs(run_names):
    pairs = sorted((run_budget(run_name), run_name) for run_name in run_names)
    budgets = [budget for budget, _ in pairs]
    if len(set(budgets)) != len(budgets):
        raise ValueError(f"Duplicate budgets in run list: {budgets}")
    return pairs


def find_common_trace_row(common, manifest_rows, root, run_names, content_run, requested_row=None):
    candidates = []
    for row_idx, item in manifest_rows.items():
        if requested_row is not None and int(row_idx) != int(requested_row):
            continue
        content_path = Path(root) / content_run / f"{item['output_prefix']}custom.mp4"
        if not content_path.is_file():
            continue
        traces = {}
        score = 0.0
        for run_name in run_names:
            trace_path = common.find_trace(root, run_name, item)
            if trace_path is None:
                break
            events = common.load_trace(trace_path)
            if not geometric.geometric_evictions_by_section(events):
                break
            traces[run_name] = (trace_path, events)
            score += geometric.trace_information_score(events)
        if len(traces) == len(run_names):
            candidates.append((score, row_idx, item, content_path, traces))
    if not candidates:
        suffix = "" if requested_row is None else f" for row {requested_row}"
        raise RuntimeError(
            "No manifest row has a common source video and complete Geometric "
            f"Coverage traces for every requested budget{suffix}"
        )
    return max(candidates, key=lambda row: (row[0], row[1]))


def reconstruct_budget_snapshots(traces, run_pairs, frames_per_section):
    all_snapshots = {}
    for budget, run_name in run_pairs:
        _, events = traces[run_name]
        run_snapshots = geometric.reconstruct_geometric_snapshots(
            events,
            budget=budget,
            frames_per_section=frames_per_section,
        )
        all_snapshots[budget] = run_snapshots
    return all_snapshots


def select_common_section(all_snapshots, requested_section=None, reference_budget=32):
    common_sections = set.intersection(
        *(set(snapshots) for snapshots in all_snapshots.values())
    )
    if not common_sections:
        raise RuntimeError("Requested budget traces have no common sections")
    if requested_section is not None:
        section_idx = int(requested_section)
        if section_idx not in common_sections:
            raise ValueError(
                f"Section {section_idx} is not present in every requested budget"
            )
        return section_idx

    if reference_budget not in all_snapshots:
        reference_budget = sorted(all_snapshots)[0]
    selected = geometric.choose_section(all_snapshots[reference_budget])
    if selected in common_sections:
        return selected
    return max(section for section in common_sections if section <= max(common_sections))


def support_curves(values, thresholds):
    values = np.asarray(values, dtype=np.float64)
    return np.asarray([np.mean(values >= threshold) for threshold in thresholds])


def coverage_rows(row_idx, scene, section_idx, universe_size, coverage):
    rows = []
    for metric, by_policy in coverage.items():
        for policy, by_budget in by_policy.items():
            for budget, values in by_budget.items():
                summary = geometric.support_summary(values)
                rows.append(
                    {
                        "row": row_idx,
                        "scene": scene,
                        "section_idx": section_idx,
                        "history_frames": universe_size,
                        "policy": policy,
                        "budget": budget,
                        "metric": metric,
                        "mean": summary["mean"],
                        "p10": summary["p10"],
                        "minimum": summary["minimum"],
                    }
                )
    return rows


def render_budget_coverage(common, coverage, output_path, content_run, row_idx, section_idx):
    plt = common.configure_matplotlib()
    from matplotlib.lines import Line2D

    budgets = sorted(coverage["camera"]["Geometric Coverage"])
    palette = plt.get_cmap("viridis")
    denominator = max(len(budgets) - 1, 1)
    colors = {
        budget: palette(0.12 + 0.78 * position / denominator)
        for position, budget in enumerate(budgets)
    }
    thresholds = np.linspace(0.0, 1.0, 201)
    titles = {
        "camera": "Direct camera-FOV support",
        "joint": "Common-source pose+DINO support",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.7), sharey=True)
    for ax, metric in zip(axes, ("camera", "joint")):
        for budget in budgets:
            color = colors[budget]
            geo_values = coverage[metric]["Geometric Coverage"][budget]
            recent_values = coverage[metric]["Recent + anchor"][budget]
            ax.plot(
                thresholds,
                support_curves(geo_values, thresholds),
                color=color,
                linewidth=2.5,
            )
            ax.plot(
                thresholds,
                support_curves(recent_values, thresholds),
                color=color,
                linewidth=1.6,
                linestyle="--",
                alpha=0.72,
            )
        ax.set_xlabel("Required support")
        ax.set_title(titles[metric], fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(color="#e7e7e7", linewidth=0.75)
        for name in ("top", "right"):
            ax.spines[name].set_visible(False)
    axes[0].set_ylabel("Fraction of historical views covered")

    budget_handles = [
        Line2D([0], [0], color=colors[budget], linewidth=2.6, label=f"B{budget}")
        for budget in budgets
    ]
    policy_handles = [
        Line2D([0], [0], color="#444444", linewidth=2.5, label="Geometric Coverage"),
        Line2D(
            [0],
            [0],
            color="#666666",
            linewidth=1.7,
            linestyle="--",
            label="Recent + anchor",
        ),
    ]
    budget_legend = axes[0].legend(
        handles=budget_handles,
        title="Actual bank budget",
        frameon=False,
        loc="lower left",
        ncol=2,
        fontsize=9,
    )
    axes[0].add_artist(budget_legend)
    axes[1].legend(handles=policy_handles, frameon=False, loc="lower left", fontsize=9)
    fig.suptitle(
        "How much historical coverage does each memory budget preserve?",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.925,
        f"Matched row {row_idx}, section {section_idx}; all visual features from {content_run}",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_csv(path, rows):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare actual Geometric Coverage banks across budgets."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    parser.add_argument("--content_run", default="baseline")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--frames_per_section", type=int, default=77)
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--section", type=int, default=None)
    parser.add_argument("--reference_budget", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--decode_timeout_sec", type=int, default=300)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    run_pairs = validate_runs(parse_list(args.runs))
    run_names = [run_name for _, run_name in run_pairs]
    common = geometric.load_common_module()
    manifest_rows = common.load_manifest(args.manifest, args.duration)
    _, row_idx, item, content_path, traces = find_common_trace_row(
        common,
        manifest_rows,
        args.root,
        run_names,
        args.content_run,
        requested_row=args.row,
    )
    all_snapshots = reconstruct_budget_snapshots(
        traces,
        run_pairs,
        frames_per_section=args.frames_per_section,
    )
    section_idx = select_common_section(
        all_snapshots,
        requested_section=args.section,
        reference_budget=args.reference_budget,
    )
    snapshots = {
        budget: run_snapshots[section_idx]
        for budget, run_snapshots in all_snapshots.items()
    }
    section_ends = {snapshot["section_end"] for snapshot in snapshots.values()}
    if len(section_ends) != 1:
        raise RuntimeError(f"Budget traces disagree on section end: {section_ends}")
    section_end = section_ends.pop()
    history_frames = list(range(section_end + 1))

    from dataset.poses import load_c2ws_from_json

    c2ws = load_c2ws_from_json(
        item["pose_path"],
        start_frame=int(item["start_frame"]),
        num_frames=int(item["num_frames"]),
    )
    print(f"Selected row {row_idx}: {item['scene']}", flush=True)
    print(f"Selected section: {section_idx}", flush=True)
    print(f"Common content: {content_path}", flush=True)
    print(f"Decoding {len(history_frames)} historical frames in one pass", flush=True)
    images = geometric.load_video_frames_single_pass(
        content_path,
        history_frames,
        timeout_sec=args.decode_timeout_sec,
    )

    policy_module = common.load_policy_module()
    print(
        f"Encoding {len(history_frames)} common-source frames with DINO on {args.device}",
        flush=True,
    )
    extractor = policy_module.VisualMemoryFeatureExtractor(device=args.device)
    dino_batch, rgb_batch = extractor.encode_pil_images(
        [images[frame_idx] for frame_idx in history_frames]
    )
    dino_features = {
        frame_idx: dino_batch[position]
        for position, frame_idx in enumerate(history_frames)
    }
    rgb_features = {
        frame_idx: rgb_batch[position]
        for position, frame_idx in enumerate(history_frames)
    }
    affinity = policy_module._slam_covisibility_affinity(
        memory_frame_indices=history_frames,
        c2ws=c2ws,
        dino_features=dino_features,
        rgb_features=rgb_features,
        self_similarity=0.0,
    )
    camera_similarity = policy_module.camera_trajectory_similarity(
        c2ws,
        history_frames,
        history_frames,
    )
    np.fill_diagonal(camera_similarity, 0.0)

    coverage = {
        "camera": {"Geometric Coverage": {}, "Recent + anchor": {}},
        "joint": {"Geometric Coverage": {}, "Recent + anchor": {}},
    }
    for budget, _ in run_pairs:
        geo_bank = snapshots[budget]["retained"]
        if len(geo_bank) > budget:
            raise RuntimeError(f"B{budget} reconstructed {len(geo_bank)} retained frames")
        recent_bank = geometric.recent_anchor_bank(history_frames, budget, anchor=0)
        for metric, similarity in (
            ("camera", camera_similarity),
            ("joint", affinity),
        ):
            coverage[metric]["Geometric Coverage"][budget] = geometric.bank_support(
                similarity,
                history_frames,
                geo_bank,
            )
            coverage[metric]["Recent + anchor"][budget] = geometric.bank_support(
                similarity,
                history_frames,
                recent_bank,
            )

    figures = args.output_dir / "figures"
    tables = args.output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    rows = coverage_rows(
        row_idx,
        item["scene"],
        section_idx,
        len(history_frames),
        coverage,
    )
    write_csv(tables / "coverage_budget_summary.csv", rows)
    output_path = render_budget_coverage(
        common,
        coverage,
        figures / f"coverage_budget_sweep_section_{section_idx}.png",
        args.content_run,
        row_idx,
        section_idx,
    )

    report = [
        "# Geometric Coverage Budget Sweep",
        "",
        f"- Matched trajectory: row `{row_idx}` (`{item['scene']}`).",
        f"- Section: `{section_idx}`; historical universe: `{len(history_frames)}` frames.",
        f"- Common visual source: `{args.content_run}`.",
        f"- Actual Geometric Coverage budgets: `{','.join(str(value) for value, _ in run_pairs)}`.",
        "",
        "Every Geometric Coverage bank is reconstructed from its real access trace. All banks are evaluated against every historical frame through the selected section. Camera poses and common-source pixels are shared across budgets; dashed controls keep the anchor and the newest frames at the same capacity.",
        "",
        "## Summary",
        "",
        "| policy | budget | metric | mean | p10 | minimum |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda value: (value["metric"], value["policy"], value["budget"])):
        report.append(
            f"| {row['policy']} | {row['budget']} | {row['metric']} | "
            f"{row['mean']:.4f} | {row['p10']:.4f} | {row['minimum']:.4f} |"
        )
    report.extend(
        [
            "",
            "## Files",
            "",
            f"- `figures/{output_path.name}`",
            "- `tables/coverage_budget_summary.csv`",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote: {output_path}")
    print(f"Wrote: {tables / 'coverage_budget_summary.csv'}")
    print(f"Wrote: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
