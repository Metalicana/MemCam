"""Does MemCam's MCE show the same "queries << budget in steady state" gap
that WorldMem's session found (median num_hist_queries vs budget, growing
gap fraction as budget increases -> queries within a saturated cluster get
tie-broken near-arbitrarily rather than discriminated)?

Reads eviction_mce_num_hist_queries directly from existing access-trace
JSONL files -- no new generation, no GPU. Steady-state approximation: only
the second half of each video's eviction records are used, to skip the
initial fill-up period before the memory bank first reaches budget.

Usage:
    python utils/check_mce_query_coverage_gap.py \
        --root ~/memcam_results/context_memory_60s \
        --runs mce_b16_lambda1_pilot:16,mce_b32_lambda1_pilot:32,mce_b64_lambda1_pilot:64,mce_b128_lambda1_pilot:128
"""

import argparse
import json
import statistics as st
from pathlib import Path


def read_num_hist_queries(trace_path):
    values = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = row.get("eviction_mce_num_hist_queries")
            if value is not None:
                values.append(int(value))
    return values


def steady_state_values(values):
    if not values:
        return []
    return values[len(values) // 2 :]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--runs", type=str, required=True,
        help="Comma-separated run_dir:budget pairs, e.g. mce_b16_lambda1_pilot:16,mce_b32_lambda1_pilot:32",
    )
    args = parser.parse_args()

    print(f"{'run':30s} {'budget':>7s} {'n_videos':>9s} {'median_queries':>15s} {'gap':>6s} {'gap_frac':>9s}")
    for spec in args.runs.split(","):
        run_name, budget_str = spec.split(":")
        budget = int(budget_str)
        run_dir = args.root / run_name / "access_traces"
        if not run_dir.is_dir():
            print(f"{run_name:30s}  -- no access_traces dir at {run_dir}")
            continue

        all_steady_values = []
        video_count = 0
        for trace_path in sorted(run_dir.glob("*.jsonl")):
            values = read_num_hist_queries(trace_path)
            steady = steady_state_values(values)
            if steady:
                all_steady_values.extend(steady)
                video_count += 1

        if not all_steady_values:
            print(f"{run_name:30s}  -- no eviction_mce_num_hist_queries values found (wrong field name, or run wasn't 'mce')")
            continue

        median_queries = st.median(all_steady_values)
        gap = budget - median_queries
        gap_frac = gap / budget
        print(
            f"{run_name:30s} {budget:7d} {video_count:9d} {median_queries:15.1f} "
            f"{gap:6.1f} {gap_frac:9.3f}"
        )

    print(
        "\nCompare against WorldMem's finding: gap/budget grew 0.31 -> 0.34 -> 0.38 -> "
        "0.39 as budget rose 16->32->64->128 (queries not keeping pace with budget, "
        "leaving a growing fraction of retained slots with no dedicated query -- "
        "decided by near-arbitrary marginal tiebreaks within a saturated cluster). "
        "If MemCam shows the same growing gap_frac trend, this is a shared, general "
        "MCE limitation, not backbone-specific."
    )


if __name__ == "__main__":
    main()
