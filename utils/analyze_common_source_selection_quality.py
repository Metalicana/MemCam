"""Compare policy-selected frame indices using one common generated video source."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.analyze_selected_memory_image_quality import (  # noqa: E402
    bootstrap_mean_interval,
    compute_requested_frame_metrics,
    exact_two_sided_sign_pvalue,
    fmt,
    load_manifest,
    load_selected_frames_by_section,
    mean,
    metric_summary,
    parse_list,
    write_csv,
)


def build_section_row(
    selection_run,
    content_run,
    item,
    section_idx,
    selected_frames,
    frame_metric_map,
):
    selected_metrics = [frame_metric_map[frame] for frame in selected_frames]
    unique_frames = sorted(set(selected_frames))
    unique_metrics = [frame_metric_map[frame] for frame in unique_frames]
    return {
        "selection_run": selection_run,
        "content_run": content_run,
        "row": int(item["_row"]),
        "scene": item["scene"],
        "start_frame": int(item["start_frame"]),
        "duration_sec": int(item["duration_sec"]),
        "section_idx": int(section_idx),
        "section_time_sec": float(
            section_idx * 76 / float(item["fps"])
        ),
        "retrievals": len(selected_frames),
        "unique_selected_frames": len(unique_frames),
        **metric_summary("selected_weighted", selected_metrics),
        **metric_summary("selected_unique", unique_metrics),
    }


def aggregate_runs(section_rows):
    grouped = defaultdict(list)
    for row in section_rows:
        grouped[row["selection_run"]].append(row)
    output = []
    for run_name, rows in sorted(grouped.items()):
        max_section = max(int(row["section_idx"]) for row in rows)
        late_start = 0.75 * max_section
        late = [row for row in rows if row["section_idx"] >= late_start]
        output.append(
            {
                "selection_run": run_name,
                "content_run": rows[0]["content_run"],
                "videos": len({row["row"] for row in rows}),
                "sections": len(rows),
                "selected_psnr_mean": mean(
                    row["selected_weighted_psnr_db_mean"] for row in rows
                ),
                "selected_ssim_mean": mean(
                    row["selected_weighted_ssim_mean"] for row in rows
                ),
                "late_selected_psnr_mean": mean(
                    row["selected_weighted_psnr_db_mean"] for row in late
                ),
                "late_selected_ssim_mean": mean(
                    row["selected_weighted_ssim_mean"] for row in late
                ),
            }
        )
    return output


def paired_rows(section_rows, reference_run):
    by_key = {
        (row["selection_run"], int(row["row"]), int(row["section_idx"])): row
        for row in section_rows
    }
    output = []
    for run_name in sorted({row["selection_run"] for row in section_rows}):
        if run_name == reference_run:
            continue
        shared = []
        for (candidate_run, row_idx, section_idx), policy in by_key.items():
            if candidate_run != run_name:
                continue
            reference = by_key.get((reference_run, row_idx, section_idx))
            if reference is not None:
                shared.append((policy, reference))
        if not shared:
            continue

        result = {
            "selection_run": run_name,
            "reference_run": reference_run,
            "content_run": shared[0][0]["content_run"],
            "paired_sections": len(shared),
            "paired_videos": len({policy["row"] for policy, _ in shared}),
        }
        for metric in ("psnr_db", "ssim"):
            field = f"selected_weighted_{metric}_mean"
            by_trajectory = defaultdict(list)
            for policy, reference in shared:
                by_trajectory[int(policy["row"])].append(
                    float(policy[field]) - float(reference[field])
                )
            trajectory_means = [mean(values) for values in by_trajectory.values()]
            wins = sum(value > 1e-12 for value in trajectory_means)
            losses = sum(value < -1e-12 for value in trajectory_means)
            ties = len(trajectory_means) - wins - losses
            ci_low, ci_high = bootstrap_mean_interval(trajectory_means)
            result[f"selected_{metric}_delta_mean"] = mean(trajectory_means)
            result[f"selected_{metric}_delta_ci_low"] = ci_low
            result[f"selected_{metric}_delta_ci_high"] = ci_high
            result[f"selected_{metric}_trajectory_wins"] = wins
            result[f"selected_{metric}_trajectory_losses"] = losses
            result[f"selected_{metric}_trajectory_ties"] = ties
            result[f"selected_{metric}_sign_pvalue"] = (
                exact_two_sided_sign_pvalue(wins, losses)
            )
        output.append(result)
    return output


def write_report(path, run_rows, comparisons, content_run, reference_run):
    lines = [
        "# Common-Source Selection Quality",
        "",
        "## Intervention",
        "",
        f"Every policy supplies only its selected frame indices. The image content for every selected index is read from `{content_run}`. Therefore differences below measure which indices a policy chooses, without comparing different generated histories.",
        "",
        "Repeated selections remain repeated because they represent repeated conditioning exposure. Higher PSNR and SSIM are better.",
        "",
        "## Summary",
        "",
        "| selection policy | content source | selected PSNR | selected SSIM | late selected PSNR | late selected SSIM |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in run_rows:
        lines.append(
            f"| {row['selection_run']} | {row['content_run']} | "
            f"{fmt(row['selected_psnr_mean'], 3)} | "
            f"{fmt(row['selected_ssim_mean'])} | "
            f"{fmt(row['late_selected_psnr_mean'], 3)} | "
            f"{fmt(row['late_selected_ssim_mean'])} |"
        )
    lines.extend(
        [
            "",
            f"## Paired Difference Versus {reference_run}",
            "",
            "Positive means that the policy selected cleaner indices than the unbounded selector from the exact same generated video. Confidence intervals resample trajectories.",
            "",
            "| selection policy | trajectories | PSNR delta | 95% CI | SSIM delta | 95% CI | PSNR wins/losses | SSIM wins/losses |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['selection_run']} | {row['paired_videos']} | "
            f"{fmt(row['selected_psnr_db_delta_mean'], 3)} | "
            f"[{fmt(row['selected_psnr_db_delta_ci_low'], 3)}, {fmt(row['selected_psnr_db_delta_ci_high'], 3)}] | "
            f"{fmt(row['selected_ssim_delta_mean'])} | "
            f"[{fmt(row['selected_ssim_delta_ci_low'])}, {fmt(row['selected_ssim_delta_ci_high'])}] | "
            f"{row['selected_psnr_db_trajectory_wins']}/{row['selected_psnr_db_trajectory_losses']} | "
            f"{row['selected_ssim_trajectory_wins']}/{row['selected_ssim_trajectory_losses']} |"
        )
    lines.extend(
        [
            "",
            "If RI or SLAM remains better here, its selected frame indices are cleaner even under a common rollout. If the advantage disappears, the earlier result mainly came from those policies having already generated cleaner histories.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection_runs", required=True)
    parser.add_argument("--content_run", default="baseline")
    parser.add_argument("--reference_run", default="baseline")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    selection_runs = parse_list(args.selection_runs)
    items = load_manifest(args.manifest, args.duration)
    section_rows = []
    for item in items:
        selections = {}
        requested = set()
        for run_name in selection_runs:
            trace = (
                args.root
                / run_name
                / "access_traces"
                / f"{item['output_prefix']}custom.jsonl"
            )
            if not trace.is_file():
                message = f"missing trace for {run_name} row {item['_row']}: {trace}"
                if args.strict:
                    raise FileNotFoundError(message)
                print(f"[skip] {message}")
                continue
            selections[run_name] = load_selected_frames_by_section(trace)
            requested.update(
                frame
                for section in selections[run_name].values()
                for frame in section
            )

        content_video = (
            args.root
            / args.content_run
            / f"{item['output_prefix']}custom.mp4"
        )
        if not content_video.is_file():
            message = f"missing common-source video for row {item['_row']}: {content_video}"
            if args.strict:
                raise FileNotFoundError(message)
            print(f"[skip] {message}")
            continue
        frame_metric_map = compute_requested_frame_metrics(
            content_video, item, requested
        )
        for run_name, by_section in selections.items():
            for section_idx, selected_frames in sorted(by_section.items()):
                section_rows.append(
                    build_section_row(
                        run_name,
                        args.content_run,
                        item,
                        section_idx,
                        selected_frames,
                        frame_metric_map,
                    )
                )
        print(
            f"row {item['_row']}: {len(frame_metric_map)} common-source frames "
            f"for {len(selections)} selectors"
        )

    if not section_rows:
        raise RuntimeError("No common-source selection rows were produced")
    run_rows = aggregate_runs(section_rows)
    comparisons = paired_rows(section_rows, args.reference_run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "section_quality.csv", section_rows)
    write_csv(args.output_dir / "run_summary.csv", run_rows)
    write_csv(args.output_dir / "paired_vs_reference.csv", comparisons)
    write_report(
        args.output_dir / "report.md",
        run_rows,
        comparisons,
        content_run=args.content_run,
        reference_run=args.reference_run,
    )
    print(f"Wrote: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
