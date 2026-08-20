"""H1 screen: what changes as the unbounded candidate pool grows?

Reads an existing tables/query_decomposition.csv already written by
analyze_retrieval_quality_decomposition.py (e.g. from
unbounded_failure_decomposition_180s) -- no new generation, no new DINO
encoding, pure CPU post-processing of a CSV that already exists on disk.

For the requested run (default: the true-unbounded ``baseline``), this checks
three quantities separately within each trajectory:

* the DINO mismatch of the frame the retriever actually selected;
* the DINO mismatch of the hindsight-best frame in full history; and
* their difference, called ``retrieval_gap``.

This split matters because the hindsight minimum can improve mechanically as
the pool grows. A growing gap is evidence that selection falls farther behind
that benchmark, but it is not evidence that the selected context itself gets
worse unless ``selected_effective_mismatch`` also increases. Pool size and
elapsed time still co-vary, so this remains a scaling screen rather than a
causal isolation of pool size.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from summarize_ri_alignment import spearman  # noqa: E402


MISMATCH_FIELDS = (
    "selected_effective_mismatch",
    "full_oracle_effective_mismatch",
    "retrieval_gap",
)


def load_query_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def paired_floats(rows, key_a, key_b):
    pairs = [(to_float(row[key_a]), to_float(row[key_b])) for row in rows]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    return xs, ys


def quantile_bin_edges(values, num_bins):
    values = np.asarray(values, dtype=np.float64)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, num_bins + 1)))
    if edges.size < 2:
        edges = np.array([values.min(), values.max() + 1.0])
    return edges


def bin_index(value, edges):
    idx = int(np.searchsorted(edges, value, side="right") - 1)
    return int(np.clip(idx, 0, len(edges) - 2))


def trajectory_key(row):
    return (
        row["row"],
        row["scene"],
        row["dataset_start_frame"],
        row["duration_sec"],
    )


def section_points(rows):
    grouped = defaultdict(list)
    for row in rows:
        count = to_float(row.get("candidate_count"))
        gap = to_float(row.get("retrieval_gap"))
        if count is None or gap is None:
            continue
        key = (*trajectory_key(row), int(row["section_idx"]))
        grouped[key].append(
            {
                "candidate_count": count,
                **{
                    field: to_float(row.get(field))
                    for field in MISMATCH_FIELDS
                },
            }
        )

    points = []
    for key, values in sorted(grouped.items()):
        row, scene, start_frame, duration_sec, section_idx = key
        point = {
            "row": row,
            "scene": scene,
            "dataset_start_frame": start_frame,
            "duration_sec": duration_sec,
            "section_idx": section_idx,
            "candidate_count": float(
                np.mean([value["candidate_count"] for value in values])
            ),
            "queries": len(values),
        }
        for field in MISMATCH_FIELDS:
            field_values = [
                value[field] for value in values if value[field] is not None
            ]
            point[field] = (
                float(np.mean(field_values)) if field_values else None
            )
        points.append(point)
    return points


def exact_two_sided_sign_pvalue(positive, negative):
    trials = int(positive) + int(negative)
    if trials == 0:
        return None
    smaller = min(int(positive), int(negative))
    tail = sum(math.comb(trials, value) for value in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2.0**trials))


def bootstrap_mean_interval(values, repeats=10000, seed=0):
    values = np.asarray([float(value) for value in values], dtype=np.float64)
    if values.size == 0:
        return None, None
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(int(seed))
    sample_indices = rng.integers(
        0, values.size, size=(int(repeats), values.size)
    )
    means = values[sample_indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_trajectories(points):
    grouped = defaultdict(list)
    for point in points:
        grouped[
            (
                point["row"],
                point["scene"],
                point["dataset_start_frame"],
                point["duration_sec"],
            )
        ].append(point)

    rows = []
    for key, trajectory_points in sorted(grouped.items()):
        trajectory_points.sort(
            key=lambda point: (point["candidate_count"], point["section_idx"])
        )
        counts = [point["candidate_count"] for point in trajectory_points]
        quartile_size = max(1, len(trajectory_points) // 4)
        row, scene, start_frame, duration_sec = key
        summary = {
            "row": row,
            "scene": scene,
            "dataset_start_frame": start_frame,
            "duration_sec": duration_sec,
            "sections": len(trajectory_points),
            "queries": sum(point["queries"] for point in trajectory_points),
            "candidate_count_min": min(counts),
            "candidate_count_max": max(counts),
        }
        for field in MISMATCH_FIELDS:
            values = [point.get(field) for point in trajectory_points]
            if any(value is None for value in values):
                continue
            early = float(np.mean(values[:quartile_size]))
            late = float(np.mean(values[-quartile_size:]))
            summary[f"spearman_pool_vs_{field}"] = spearman(counts, values)
            summary[f"early_quartile_{field}"] = early
            summary[f"late_quartile_{field}"] = late
            summary[f"late_minus_early_{field}"] = late - early
            summary[f"linear_slope_{field}"] = (
                float(np.polyfit(counts, values, 1)[0])
                if len(set(counts)) >= 2
                else None
            )

        # Preserve the original column names for downstream users of this CSV.
        if "spearman_pool_vs_retrieval_gap" in summary:
            summary["linear_slope"] = summary["linear_slope_retrieval_gap"]
            summary["early_quartile_gap"] = summary[
                "early_quartile_retrieval_gap"
            ]
            summary["late_quartile_gap"] = summary[
                "late_quartile_retrieval_gap"
            ]
            summary["late_minus_early_gap"] = summary[
                "late_minus_early_retrieval_gap"
            ]
        rows.append(summary)
    return rows


def write_csv(path, rows):
    fields = list(rows[0].keys()) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query_csv",
        type=Path,
        required=True,
        help="Path to an existing tables/query_decomposition.csv "
        "(from analyze_retrieval_quality_decomposition.py).",
    )
    parser.add_argument("--run_name", type=str, default="baseline")
    parser.add_argument("--num_bins", type=int, default=8)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_query_rows(args.query_csv)
    run_rows = [row for row in rows if row["run_name"] == args.run_name]
    if not run_rows:
        available = sorted({row["run_name"] for row in rows})
        raise RuntimeError(
            f"No rows for run_name={args.run_name!r} in {args.query_csv} "
            f"(available: {available})"
        )

    retention_gaps = [to_float(row["retention_gap"]) for row in run_rows]
    retention_gaps = [g for g in retention_gaps if g is not None]
    retention_violations = sum(1 for g in retention_gaps if g > 1e-6)
    retention_violation_rate = (
        retention_violations / len(retention_gaps) if retention_gaps else None
    )

    pooled_correlations = {}
    for field in MISMATCH_FIELDS:
        count_xs, count_ys = paired_floats(run_rows, "candidate_count", field)
        section_xs, section_ys = paired_floats(run_rows, "section_idx", field)
        pooled_correlations[field] = {
            "candidate_count": spearman(count_xs, count_ys),
            "section_idx": spearman(section_xs, section_ys),
        }
    rho_count = pooled_correlations["retrieval_gap"]["candidate_count"]
    rho_section = pooled_correlations["retrieval_gap"]["section_idx"]

    points = section_points(run_rows)
    if not points:
        raise RuntimeError(
            f"No candidate_count/retrieval_gap section points for {args.run_name!r}"
        )
    trajectory_rows = summarize_trajectories(points)
    trajectory_rhos = [
        row["spearman_pool_vs_retrieval_gap"]
        for row in trajectory_rows
        if row["spearman_pool_vs_retrieval_gap"] is not None
    ]
    positive_rhos = sum(value > 0 for value in trajectory_rhos)
    negative_rhos = sum(value < 0 for value in trajectory_rhos)
    zero_rhos = sum(value == 0 for value in trajectory_rhos)
    sign_pvalue = exact_two_sided_sign_pvalue(positive_rhos, negative_rhos)
    late_deltas_by_field = {
        field: [
            row[f"late_minus_early_{field}"]
            for row in trajectory_rows
            if row.get(f"late_minus_early_{field}") is not None
        ]
        for field in MISMATCH_FIELDS
    }
    late_intervals = {
        field: bootstrap_mean_interval(values)
        for field, values in late_deltas_by_field.items()
    }
    late_deltas = late_deltas_by_field["retrieval_gap"]
    late_ci_low, late_ci_high = late_intervals["retrieval_gap"]

    section_counts = [point["candidate_count"] for point in points]
    edges = quantile_bin_edges(section_counts, args.num_bins)
    bins = [[] for _ in range(len(edges) - 1)]
    for point in points:
        bins[bin_index(point["candidate_count"], edges)].append(point)

    bin_rows = []
    for bin_idx, bin_points in enumerate(bins):
        if not bin_points:
            continue
        bin_row = {
            "bin": bin_idx,
            "candidate_count_low": edges[bin_idx],
            "candidate_count_high": edges[bin_idx + 1],
            "trajectory_sections": len(bin_points),
        }
        for field in MISMATCH_FIELDS:
            values = [
                point[field]
                for point in bin_points
                if point.get(field) is not None
            ]
            bin_row[f"mean_{field}"] = (
                float(np.mean(values)) if values else None
            )
        bin_row["error_rate"] = sum(
            1 for point in bin_points if point["retrieval_gap"] > 1e-6
        ) / len(bin_points)
        bin_rows.append(bin_row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / f"pool_growth_scaling_{args.run_name}.csv"
    trajectory_csv = (
        args.output_dir / f"pool_growth_per_trajectory_{args.run_name}.csv"
    )
    summary_json = args.output_dir / f"pool_growth_summary_{args.run_name}.json"
    write_csv(out_csv, bin_rows)
    write_csv(trajectory_csv, trajectory_rows)
    summary = {
        "run_name": args.run_name,
        "queries": len(run_rows),
        "trajectory_sections": len(points),
        "trajectories": len(trajectory_rows),
        "retention_gap_violations": retention_violations,
        "pooled_query_spearman_candidate_count": rho_count,
        "pooled_query_spearman_section_idx": rho_section,
        "pooled_correlations": pooled_correlations,
        "trajectory_positive_rho": positive_rhos,
        "trajectory_negative_rho": negative_rhos,
        "trajectory_zero_rho": zero_rhos,
        "trajectory_sign_test_pvalue": sign_pvalue,
        "mean_late_minus_early_gap": (
            float(np.mean(late_deltas)) if late_deltas else None
        ),
        "mean_late_minus_early_gap_ci_low": late_ci_low,
        "mean_late_minus_early_gap_ci_high": late_ci_high,
        "late_minus_early": {
            field: {
                "mean": float(np.mean(values)) if values else None,
                "ci_low": late_intervals[field][0],
                "ci_high": late_intervals[field][1],
            }
            for field, values in late_deltas_by_field.items()
        },
        "gap_growth_identity": (
            "retrieval_gap change = selected mismatch change - hindsight-best "
            "mismatch change"
        ),
        "interpretation_limit": (
            "The hindsight-best mismatch can decrease mechanically as the pool "
            "grows. Candidate count and elapsed time also co-vary. Only growth "
            "in selected_effective_mismatch shows that the selected context "
            "itself becomes less target-matched under DINO."
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"H1: pool-growth scaling for run_name={args.run_name!r}")
    print(f"Queries analyzed: {len(run_rows)}")
    print(f"Trajectory-sections: {len(points)} across {len(trajectory_rows)} trajectories")
    retention_rate_text = (
        f"{retention_violation_rate:.2%}"
        if retention_violation_rate is not None
        else "NA"
    )
    print(
        f"Retention-gap violations (>1e-6): {retention_violations}/{len(retention_gaps)} "
        f"({retention_rate_text}) -- should be ~0 for a run whose bank is the "
        "full history by construction"
    )
    print(f"Pooled query Spearman(candidate_count, retrieval_gap) = {rho_count}")
    print(f"Pooled query Spearman(section_idx, retrieval_gap)     = {rho_section}")
    for field in (
        "selected_effective_mismatch",
        "full_oracle_effective_mismatch",
    ):
        print(
            f"Pooled query Spearman(candidate_count, {field}) = "
            f"{pooled_correlations[field]['candidate_count']}"
        )
    print(
        "Per-trajectory pool/gap trend: "
        f"positive={positive_rhos}, negative={negative_rhos}, zero={zero_rhos}, "
        f"exact sign-test p={sign_pvalue}"
    )
    if late_deltas:
        print(
            "Mean late-minus-early retrieval gap: "
            f"{np.mean(late_deltas):.6f} "
            f"(trajectory bootstrap 95% CI {late_ci_low:.6f}, {late_ci_high:.6f})"
        )
    print("Late-minus-early decomposition (trajectory means):")
    for field in MISMATCH_FIELDS:
        values = late_deltas_by_field[field]
        ci_low, ci_high = late_intervals[field]
        if values:
            print(
                f"  {field}: {np.mean(values):+.6f} "
                f"(95% CI {ci_low:+.6f}, {ci_high:+.6f})"
            )
    print(
        "Identity: gap change = selected mismatch change - hindsight-best "
        "mismatch change."
    )
    print(
        "Interpretation: if selected mismatch is flat while the hindsight-best "
        "mismatch falls, the growing gap is mainly an opportunity-set artifact."
    )
    print()
    print(
        f"{'bin':>3}  {'candidate_count range':<24}{'n sections':>12}  "
        f"{'selected':>10}  {'hindsight':>10}  {'gap':>10}"
    )
    for row in bin_rows:
        count_range = f"[{row['candidate_count_low']:.0f}, {row['candidate_count_high']:.0f})"
        print(
            f"{row['bin']:>3}  {count_range:<24}{row['trajectory_sections']:>12}  "
            f"{row['mean_selected_effective_mismatch']:>10.4f}  "
            f"{row['mean_full_oracle_effective_mismatch']:>10.4f}  "
            f"{row['mean_retrieval_gap']:>10.4f}"
        )
    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {trajectory_csv}")
    print(f"Wrote: {summary_json}")


if __name__ == "__main__":
    main()
