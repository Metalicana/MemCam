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
import math
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


def values_by_section(rows, run_name, metric):
    values = {}
    for row in rows:
        if row["run_name"] != run_name:
            continue
        value = to_float(row.get(metric))
        if value is None:
            continue
        key = (*trajectory_key(row), int(row["section_idx"]))
        values[key] = value
    return values


def paired_by_trajectory(rows, left_run, right_run, metric):
    left = values_by_section(rows, left_run, metric)
    right = values_by_section(rows, right_run, metric)
    shared_sections = sorted(set(left) & set(right))
    grouped = defaultdict(list)
    for section_key in shared_sections:
        trajectory = section_key[:-1]
        grouped[trajectory].append((left[section_key], right[section_key]))

    paired = {}
    for trajectory, values in grouped.items():
        paired[trajectory] = {
            "left": sum(value[0] for value in values) / len(values),
            "right": sum(value[1] for value in values) / len(values),
            "delta": sum(value[0] - value[1] for value in values) / len(values),
            "shared_sections": len(values),
        }
    return paired


def exact_two_sided_sign_pvalue(wins, losses):
    trials = int(wins) + int(losses)
    if trials == 0:
        return None
    smaller = min(int(wins), int(losses))
    tail = sum(math.comb(trials, value) for value in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2.0**trials))


def sign_test(paired, tolerance=1e-12):
    wins = sum(values["delta"] < -tolerance for values in paired.values())
    losses = sum(values["delta"] > tolerance for values in paired.values())
    ties = len(paired) - wins - losses
    return {
        "trajectories": len(paired),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "pvalue": exact_two_sided_sign_pvalue(wins, losses),
        "mean_delta": (
            sum(values["delta"] for values in paired.values()) / len(paired)
            if paired
            else None
        ),
    }


def print_sign_test(label, result, left_name, right_name):
    n = result["trajectories"]
    if n == 0:
        print(f"{label}: no shared trajectories between {left_name} and {right_name}")
        return
    print(
        f"{label}: {left_name} wins {result['wins']}, loses {result['losses']}, "
        f"ties {result['ties']} across {n} trajectories vs {right_name}; "
        f"mean {left_name}-{right_name} delta={result['mean_delta']:+.6f}; "
        f"exact sign-test p={result['pvalue']}"
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

    def pair(metric, left_run, right_run):
        result = paired_by_trajectory(rows, left_run, right_run, metric)
        if not result:
            print(
                f"  [warn] no matched {metric} sections for {left_run!r} vs "
                f"{right_run!r} (available: {available})"
            )
        return result

    retention_pair = pair(
        "retention_gap", args.retention_run, args.retrieval_run
    )
    retrieval_pair = pair(
        "retrieval_gap", args.retrieval_run, args.retention_run
    )

    print(f"H2 per-trajectory sign test: {args.retention_run} vs {args.retrieval_run}")
    print()

    retention_result = sign_test(retention_pair)
    print_sign_test(
        "retention_gap (lower is better)",
        retention_result,
        args.retention_run,
        args.retrieval_run,
    )

    retrieval_result = sign_test(retrieval_pair)
    print_sign_test(
        "retrieval_gap (lower is better)",
        retrieval_result,
        args.retrieval_run,
        args.retention_run,
    )

    # Context: does each policy actually beat its "natural" un-optimized
    # opponent on the mechanism it targets (RI vs FIFO on retention, SLAM vs
    # unbounded on retrieval)? If a policy doesn't even clear that bar
    # per-trajectory, the pooled-average story is on even weaker ground.
    if args.fifo_run:
        fifo_pair = pair("retention_gap", args.retention_run, args.fifo_run)
        fifo_result = sign_test(fifo_pair)
        print_sign_test(
            "retention_gap vs fifo",
            fifo_result,
            args.retention_run,
            args.fifo_run,
        )
    if args.baseline_run:
        baseline_pair = pair(
            "retrieval_gap", args.retrieval_run, args.baseline_run
        )
        baseline_result = sign_test(baseline_pair)
        print_sign_test(
            "retrieval_gap vs baseline (unbounded)",
            baseline_result,
            args.retrieval_run,
            args.baseline_run,
        )

    if args.output_csv is not None:
        shared_keys = sorted(set(retention_pair) & set(retrieval_pair))
        out_rows = []
        for key in shared_keys:
            row, scene, start_frame, duration_sec = key
            retention_values = retention_pair[key]
            # retrieval_pair is oriented retrieval_run minus retention_run.
            retrieval_values = retrieval_pair[key]
            out_rows.append(
                {
                    "row": row,
                    "scene": scene,
                    "dataset_start_frame": start_frame,
                    "duration_sec": duration_sec,
                    "retention_shared_sections": retention_values["shared_sections"],
                    f"{args.retention_run}_retention_gap": retention_values["left"],
                    f"{args.retrieval_run}_retention_gap": retention_values["right"],
                    "retention_delta": retention_values["delta"],
                    "retrieval_shared_sections": retrieval_values["shared_sections"],
                    f"{args.retrieval_run}_retrieval_gap": retrieval_values["left"],
                    f"{args.retention_run}_retrieval_gap": retrieval_values["right"],
                    "retrieval_delta": retrieval_values["delta"],
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
