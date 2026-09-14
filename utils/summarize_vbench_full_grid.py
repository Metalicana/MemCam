"""Validate and summarize exact-cohort VBench and VBench-Long results."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

try:
    from audit_metric_coverage import DIMENSIONS, bench_details, finite, valid_bench_score
    from stage_matched_video_cohort import DEFAULT_RUNS, load_cohort
except ModuleNotFoundError:
    from utils.audit_metric_coverage import DIMENSIONS, bench_details, finite, valid_bench_score
    from utils.stage_matched_video_cohort import DEFAULT_RUNS, load_cohort


def expected_names(manifest: Path, duration: int, expected_videos: int):
    cohort = load_cohort(manifest, duration)
    names = {item["output_prefix"] + "custom.mp4" for _, item in cohort}
    if len(cohort) != expected_videos or len(names) != expected_videos:
        raise ValueError(
            f"Expected {expected_videos} distinct manifest videos at {duration}s, "
            f"got {len(cohort)} rows and {len(names)} names"
        )
    return names


def normalized_score(dimension, value):
    score = float(value)
    return score / 100.0 if dimension == "imaging_quality" else score


def validate_result(path: Path, expected):
    payload = json.loads(path.read_text())
    scores = {}
    for dimension in DIMENSIONS:
        details = bench_details(payload.get(dimension))
        if details is None:
            raise ValueError(f"{dimension}: missing detailed results")
        names = [Path(str(row.get("video_path", ""))).name for row in details]
        if len(names) != len(set(names)):
            raise ValueError(f"{dimension}: duplicate video records")
        if set(names) != expected:
            raise ValueError(
                f"{dimension}: cohort mismatch; missing={sorted(expected - set(names))}, "
                f"extra={sorted(set(names) - expected)}"
            )
        values = []
        for row in details:
            value = row.get("video_results")
            if not valid_bench_score(dimension, value):
                raise ValueError(f"{dimension}: invalid score for {row.get('video_path')}")
            values.append(normalized_score(dimension, value))
        aggregate = payload[dimension][0]
        if not finite(aggregate):
            raise ValueError(f"{dimension}: invalid stored aggregate")
        scores[dimension] = {
            "mean": mean(values),
            "stored_aggregate": float(aggregate),
            "n": len(values),
        }
    return scores


def find_exact_result(directory: Path, expected):
    failures = []
    candidates = sorted(
        directory.glob("*_eval_results.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    for path in candidates:
        try:
            return path, validate_result(path, expected)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(f"{path}: {exc}")
    detail = "; ".join(failures) if failures else "no result JSON found"
    raise ValueError(f"No exact-cohort result in {directory}: {detail}")


def policy_fields(run):
    if run == "baseline":
        return "Unbounded", None
    for prefix, label in (
        ("fifo_b", "FIFO"),
        ("ri_b", "RI"),
        ("slam_b", "GeoCov"),
        ("kcenter_b", "K-center"),
        ("mce_b", "MCE"),
    ):
        if run.startswith(prefix):
            budget = int(run[len(prefix):].split("_", 1)[0])
            return label, budget
    return run, None


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--standard-root", type=Path)
    parser.add_argument("--long-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--expected-videos", type=int, default=15)
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    args = parser.parse_args()
    roots = {name: root for name, root in (
        ("VBench", args.standard_root), ("VBench-Long", args.long_root)
    ) if root is not None}
    if not roots:
        parser.error("Provide --standard-root and/or --long-root")

    expected = expected_names(args.manifest, args.duration, args.expected_videos)
    runs = [run.strip() for run in args.runs.split(",") if run.strip()]
    rows = []
    issues = []
    for evaluator, root in roots.items():
        for run in runs:
            try:
                source, scores = find_exact_result(root / run, expected)
                policy, budget = policy_fields(run)
                row = {
                    "evaluator": evaluator,
                    "run_name": run,
                    "policy": policy,
                    "budget": budget,
                    "videos": args.expected_videos,
                    **{dimension: scores[dimension]["mean"] for dimension in DIMENSIONS},
                    "source_json": str(source),
                }
                rows.append(row)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                issues.append(f"{evaluator}/{run}: {exc}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        write_csv(args.output_dir / "summary.csv", rows)
    report = ["# Exact-15 VBench Summary", ""]
    for evaluator in roots:
        report += [f"## {evaluator}", "",
                   "| Policy | B | N | Subject | Background | Motion | Dynamic | Aesthetic | Imaging |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in [item for item in rows if item["evaluator"] == evaluator]:
            budget = "-" if row["budget"] is None else str(row["budget"])
            values = " | ".join(f"{row[d]:.6f}" for d in DIMENSIONS)
            report.append(f"| {row['policy']} | {budget} | {row['videos']} | {values} |")
        report.append("")
    if issues:
        report += ["## Issues", ""] + [f"- {issue}" for issue in issues] + [""]
    (args.output_dir / "summary.md").write_text("\n".join(report))
    (args.output_dir / "summary.json").write_text(json.dumps({
        "duration": args.duration,
        "expected_videos": sorted(expected),
        "results": rows,
        "issues": issues,
    }, indent=2) + "\n")
    print("\n".join(report))
    print(f"Wrote: {args.output_dir}")
    if issues:
        raise SystemExit(f"Incomplete or mismatched results: {len(issues)} cells")


if __name__ == "__main__":
    main()
