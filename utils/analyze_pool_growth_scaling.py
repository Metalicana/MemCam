"""H1: does unbounded memory fail because selector precision decreases as the
candidate pool grows?

Reads an existing tables/query_decomposition.csv already written by
analyze_retrieval_quality_decomposition.py (e.g. from
unbounded_failure_decomposition_180s) -- no new generation, no new DINO
encoding, pure CPU post-processing of a CSV that already exists on disk.

For the requested run (default: the true-unbounded "baseline" run, whose
bank is the full history by construction, so retention_gap should already be
~0 everywhere and any avoidable error must show up as retrieval_gap), this
checks whether retrieval_gap trends upward with candidate_count (pool size)
and with section_idx (a monotone proxy for elapsed time / pool growth).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from summarize_ri_alignment import spearman  # noqa: E402


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

    count_xs, count_ys = paired_floats(run_rows, "candidate_count", "retrieval_gap")
    section_xs, section_ys = paired_floats(run_rows, "section_idx", "retrieval_gap")
    rho_count = spearman(count_xs, count_ys)
    rho_section = spearman(section_xs, section_ys)

    edges = quantile_bin_edges(count_xs, args.num_bins)
    bins = [[] for _ in range(len(edges) - 1)]
    for count, gap in zip(count_xs, count_ys):
        bins[bin_index(count, edges)].append(gap)

    bin_rows = []
    for bin_idx, gaps in enumerate(bins):
        if not gaps:
            continue
        bin_rows.append(
            {
                "bin": bin_idx,
                "candidate_count_low": edges[bin_idx],
                "candidate_count_high": edges[bin_idx + 1],
                "queries": len(gaps),
                "mean_retrieval_gap": sum(gaps) / len(gaps),
                "error_rate": sum(1 for g in gaps if g > 1e-6) / len(gaps),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / f"pool_growth_scaling_{args.run_name}.csv"
    fieldnames = list(bin_rows[0].keys()) if bin_rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if bin_rows:
            writer.writeheader()
            writer.writerows(bin_rows)

    print(f"H1: pool-growth scaling for run_name={args.run_name!r}")
    print(f"Queries analyzed: {len(run_rows)}")
    print(
        f"Retention-gap violations (>1e-6): {retention_violations}/{len(retention_gaps)} "
        f"({retention_violations / len(retention_gaps):.2%}) -- should be ~0 for a "
        "run whose bank is the full history by construction"
    )
    print(f"Spearman(candidate_count, retrieval_gap) = {rho_count}")
    print(f"Spearman(section_idx, retrieval_gap)     = {rho_section}")
    print()
    print(f"{'bin':>3}  {'candidate_count range':<24}{'n':>6}  {'mean_retrieval_gap':>18}  {'error_rate':>10}")
    for row in bin_rows:
        count_range = f"[{row['candidate_count_low']:.0f}, {row['candidate_count_high']:.0f})"
        print(
            f"{row['bin']:>3}  {count_range:<24}{row['queries']:>6}  "
            f"{row['mean_retrieval_gap']:>18.4f}  {row['error_rate']:>10.2%}"
        )
    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()
