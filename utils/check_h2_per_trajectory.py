"""H2: do RI and SLAM actually reduce different failure modes, per trajectory
-- not just in one pooled average table?

Reads an existing tables/section_summary.csv already written by
analyze_retrieval_quality_decomposition.py (grouped by run_name, row, scene,
dataset_start_frame, duration_sec, section_idx) -- no new generation or DINO
encoding, pure CPU post-processing of a CSV that already exists on disk.

For each of the two named runs, aggregates retention_gap/retrieval_gap up to
one number per (row, scene, dataset_start_frame, duration_sec) -- i.e. per
real trajectory -- then reports, trajectory by trajectory, whether
retention_run's retention_gap < retrieval_run's, and whether
retrieval_run's retrieval_gap < retention_run's. A claim that only holds in
a pooled average and not in most individual trajectories is not the same
claim as "RI reduces retention loss / SLAM reduces retrieval loss" -- this
is the sign test that tells the two apart.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def trajectory_key(row):
    return (row["row"], row["scene"], row["dataset_start_frame"], row["duration_sec"])


def aggregate_by_trajectory(rows, run_name, metric):
    grouped = defaultdict(list)
    for row in rows:
        if row["run_name"] != run_name:
            continue
        value = to_float(row.get(metric))
        if value is None:
            continue
        grouped[trajectory_key(row)].append(value)
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def sign_test(left_by_traj, right_by_traj, comparison):
    """comparison(left_value, right_value) -> bool "left wins this trajectory"."""
    shared = sorted(set(left_by_traj) & set(right_by_traj))
    wins = sum(1 for key in shared if comparison(left_by_traj[key], right_by_traj[key]))
    return shared, wins


def print_sign_test(label, shared, wins, left_name, right_name):
    n = len(shared)
    if n == 0:
        print(f"{label}: no shared trajectories between {left_name} and {right_name}")
        return
    print(
        f"{label}: {left_name} wins {wins}/{n} trajectories "
        f"({wins / n:.0%}) vs {right_name}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section_csv",
        type=Path,
        required=True,
        help="Path to an existing tables/section_summary.csv "
        "(from analyze_retrieval_quality_decomposition.py).",
    )
    parser.add_argument("--retention_run", type=str, default="ri_b32_dino_rgb")
    parser.add_argument("--retrieval_run", type=str, default="slam_b32_covisibility")
    parser.add_argument("--baseline_run", type=str, default="baseline")
    parser.add_argument("--fifo_run", type=str, default="fifo_b32")
    parser.add_argument("--output_csv", type=Path, default=None)
    args = parser.parse_args()

    rows = load_rows(args.section_csv)
    available = sorted({row["run_name"] for row in rows})

    def by_run(metric, run_name):
        result = aggregate_by_trajectory(rows, run_name, metric)
        if not result:
            print(f"  [warn] no {metric} rows for run_name={run_name!r} (available: {available})")
        return result

    retention_of_retention_run = by_run("retention_gap", args.retention_run)
    retention_of_retrieval_run = by_run("retention_gap", args.retrieval_run)
    retrieval_of_retention_run = by_run("retrieval_gap", args.retention_run)
    retrieval_of_retrieval_run = by_run("retrieval_gap", args.retrieval_run)

    print(f"H2 per-trajectory sign test: {args.retention_run} vs {args.retrieval_run}")
    print()

    shared, wins = sign_test(
        retention_of_retention_run,
        retention_of_retrieval_run,
        lambda a, b: a < b,
    )
    print_sign_test(
        "retention_gap (lower is better)", shared, wins, args.retention_run, args.retrieval_run
    )

    shared2, wins2 = sign_test(
        retrieval_of_retrieval_run,
        retrieval_of_retention_run,
        lambda a, b: a < b,
    )
    print_sign_test(
        "retrieval_gap (lower is better)", shared2, wins2, args.retrieval_run, args.retention_run
    )

    # Context: does each policy actually beat its "natural" un-optimized
    # opponent on the mechanism it targets (RI vs FIFO on retention, SLAM vs
    # unbounded on retrieval)? If a policy doesn't even clear that bar
    # per-trajectory, the pooled-average story is on even weaker ground.
    if args.fifo_run:
        retention_of_fifo = by_run("retention_gap", args.fifo_run)
        shared3, wins3 = sign_test(
            retention_of_retention_run, retention_of_fifo, lambda a, b: a < b
        )
        print_sign_test(
            "retention_gap vs fifo", shared3, wins3, args.retention_run, args.fifo_run
        )
    if args.baseline_run:
        retrieval_of_baseline = by_run("retrieval_gap", args.baseline_run)
        shared4, wins4 = sign_test(
            retrieval_of_retrieval_run, retrieval_of_baseline, lambda a, b: a < b
        )
        print_sign_test(
            "retrieval_gap vs baseline (unbounded)", shared4, wins4, args.retrieval_run, args.baseline_run
        )

    if args.output_csv is not None:
        shared_keys = sorted(
            set(retention_of_retention_run)
            & set(retention_of_retrieval_run)
            & set(retrieval_of_retention_run)
            & set(retrieval_of_retrieval_run)
        )
        out_rows = []
        for key in shared_keys:
            row, scene, start_frame, duration_sec = key
            out_rows.append(
                {
                    "row": row,
                    "scene": scene,
                    "dataset_start_frame": start_frame,
                    "duration_sec": duration_sec,
                    f"{args.retention_run}_retention_gap": retention_of_retention_run[key],
                    f"{args.retrieval_run}_retention_gap": retention_of_retrieval_run[key],
                    f"{args.retention_run}_retrieval_gap": retrieval_of_retention_run[key],
                    f"{args.retrieval_run}_retrieval_gap": retrieval_of_retrieval_run[key],
                }
            )
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()) if out_rows else [])
            if out_rows:
                writer.writeheader()
                writer.writerows(out_rows)
        print(f"\nWrote: {args.output_csv}")


if __name__ == "__main__":
    main()
