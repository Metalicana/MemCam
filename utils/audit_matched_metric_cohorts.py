"""CPU-only exact-cohort audit. Reads results; never runs an evaluator.

EXACT means an exact-cohort result file exists, not that a manuscript number
was taken from it or that its metric settings match other files.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from audit_metric_coverage import DIMENSIONS, bench_details, finite, valid_bench_score
from collect_matched_video_metrics import collect


PATTERNS = ("fifo_b{}", "ri_b{}_dino_rgb", "slam_b{}_covisibility",
            "kcenter_b{}", "mce_b{}_lambda1_pilot")
RUNS = ["baseline"] + [pattern.format(b) for pattern in PATTERNS for b in (16, 32, 64, 128)]


def identity(row):
    path = Path(str(row.get("output", "")))
    return path.parent.name, path.name


def cohort_record(source, names, expected, problems=None, **metadata):
    counts = Counter(names)
    names = set(counts)
    issues = list(problems or [])
    if any(n > 1 for n in counts.values()):
        issues.append("duplicate video records")
    missing, extra = sorted(expected - names), sorted(names - expected)
    state = "EXACT" if not missing and not extra else "SUPERSET" if not missing else "PARTIAL"
    if issues:
        state = "CHECK"
    return {"source": str(source), "state": state, "covered": sorted(expected & names),
            "missing": missing, "extra": extra, "issues": issues, **metadata}


def scan_standard(root, expected, frame_count, warnings):
    records = {run: {metric: [] for metric in ("LPIPS", "FVD", "VB6")} for run in RUNS}
    for path in sorted(root.rglob("metrics.jsonl")):
        try:
            data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            summary_path = path.with_name("summary.json")
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            config = summary.get("metric_config", {})
            rows = [r for r in data if str(r.get("duration_sec")) == "60"]
            cap = config.get("max_frames")
            config_issues = []
            if "max_frames" not in config:
                config_issues.append("full-duration max_frames setting not recorded")
            elif cap is not None and int(cap) > 0 and int(cap) < frame_count:
                config_issues.append(f"prefix evaluation: max_frames={cap}")
            for run in RUNS:
                own = [r for r in rows if identity(r)[0] == run]
                if not own:
                    continue
                lpips = [r for r in own if r.get("status") == "completed" and finite(r.get("lpips_alex"))]
                if lpips:
                    records[run]["LPIPS"].append(cohort_record(
                        path, [identity(r)[1] for r in lpips], expected, config_issues,
                        metric_config=config, scope="valid per-video LPIPS records; aggregate not assumed"))
                fvd = summary.get("by_duration", {}).get("60", {})
                if not finite(fvd.get("fvd")):
                    continue
                # FVD belongs to the entire duration group, never a per-run slice of it.
                members = [r for r in rows if r.get("status") in ("completed", "short_video")]
                issues = list(config_issues)
                if any(identity(r)[0] != run for r in members):
                    issues.append("FVD duration group includes other policies")
                if any(r.get("status") == "short_video" for r in members):
                    issues.append("FVD includes short videos")
                if fvd.get("completed_or_short") != len(members):
                    issues.append("FVD reported count missing or inconsistent with member records")
                records[run]["FVD"].append(cohort_record(
                    summary_path, [identity(r)[1] for r in members], expected, issues,
                    metric_config=config, value=fvd["fvd"], group=fvd))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            warnings.append(f"{path}: {exc}")

    for path in sorted(root.rglob("*_eval_results.json")):
        if "long" in str(path.parent).lower():
            continue
        try:
            payload = json.loads(path.read_text())
            by_run = {}
            global_members = {}
            for dim in DIMENSIONS:
                details = bench_details(payload.get(dim))
                if details is None:
                    continue
                global_members[dim] = [str(r.get("video_path", "")) for r in details]
                for row in details:
                    video = Path(str(row.get("video_path", "")))
                    if video.parent.name not in records:
                        continue
                    bucket = by_run.setdefault(video.parent.name, {d: [] for d in DIMENSIONS})
                    bucket[dim].append(row)
            for run, dimensions in by_run.items():
                checks = {}
                issues = []
                for dim, rows in dimensions.items():
                    names = [Path(r["video_path"]).name for r in rows
                             if valid_bench_score(dim, r.get("video_results"))]
                    checks[dim] = cohort_record(path, names, expected)
                    issues.extend(f"{dim}: {issue}" for issue in checks[dim]["issues"])
                    if len(names) != len(rows):
                        issues.append(f"{dim}: invalid scores")
                    if any(Path(p).parent.name != run for p in global_members.get(dim, [])):
                        issues.append(f"{dim}: top-level aggregate includes other policies")
                covered = set.intersection(*(set(c["covered"]) for c in checks.values()))
                extras = set.union(*(set(c["extra"]) for c in checks.values()))
                records[run]["VB6"].append(cohort_record(
                    path, sorted(covered | extras), expected, issues, dimensions=checks,
                    scope="all six dimensions; SUPERSET requires extracting the manifest subset, not using the stored aggregate"))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            warnings.append(f"{path}: {exc}")
    return records


def cell(records, expected):
    for state in ("EXACT", "SUPERSET"):
        if any(r["state"] == state for r in records):
            return state
    if not records:
        return "NONE"
    valid = [r for r in records if r["state"] != "CHECK"]
    covered = set().union(*(set(r["covered"]) for r in valid))
    return f"UNION {len(covered)}/{len(expected)}" if valid else "CHECK"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "memcam_results")
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("Results root does not exist")
    manifest = {i: json.loads(line) for i, line in enumerate(args.manifest.read_text().splitlines()) if line.strip()}
    cohort = {i: r for i, r in manifest.items() if int(r["duration_sec"]) == 60}
    expected = {r["output_prefix"] + "custom.mp4" for r in cohort.values()}
    if len(cohort) != 15 or len(expected) != 15:
        parser.error("Manifest must contain exactly 15 distinct 60-second entries")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    warnings = []
    records = scan_standard(args.root, expected, max(int(r["num_frames"]) for r in cohort.values()), warnings)
    suite_root = args.root / "matched15_metrics_60s"
    suites = collect(suite_root) if suite_root.is_dir() else {"coverage": [], "per_video": [], "warnings": []}
    warnings.extend(suites["warnings"])
    suite_checks = {}
    for record in suites["coverage"]:
        issues = list(record["issues"])
        if "suite" in record:
            source_manifest = Path(record["suite"]) / "manifest.jsonl"
            try:
                saved = {i: json.loads(line) for i, line in enumerate(source_manifest.read_text().splitlines()) if line.strip()}
                if {i: r for i, r in saved.items() if int(r["duration_sec"]) == 60} != cohort:
                    issues.append("suite manifest differs from the supplied canonical manifest")
            except (OSError, ValueError, KeyError) as exc:
                issues.append(str(exc))
        names = [r["video"] for r in suites["per_video"]
                 if r["run"] == record["run"] and r["evaluator"] == record["evaluator"]]
        suite_checks[record["run"], record["evaluator"]] = cohort_record(
            record.get("suite", "missing suite"), names, expected, issues,
            valid=len(names))

    lines = ["# Exact 15-video cohort audit (60 seconds)", "",
             "| Run | Files | LPIPS | FVD | VBench six | VBench-Long | CUT3R |",
             "|---|---:|---|---|---|---|---|"]
    for run in RUNS:
        present = {p.name for p in (args.root / "context_memory_60s" / run).glob("*_custom.mp4")
                   if p.is_file() and p.stat().st_size > 0}
        metric_cells = [cell(records[run][metric], expected) for metric in ("LPIPS", "FVD", "VB6")]
        for evaluator in ("vbench-long", "cut3r"):
            check = suite_checks.get((run, evaluator))
            metric_cells.append(("EXACT" if check["state"] == "EXACT" else f"{check['valid']}/15 CHECK")
                                if check else "NOT AUDITED")
        lines.append(f"| {run} | {len(expected & present)}/15 | " + " | ".join(metric_cells) + " |")
    lines += ["", "EXACT: at least one exact-cohort file exists. SUPERSET: all 15 plus extras; do not use its stored aggregate.",
              "UNION: coverage spread across incomplete files, not a verified matched aggregate. CHECK: inspect JSON issues.",
              "Long/CUT3R use the latest matched-suite attempts only; other budgets' legacy camera/Long outputs are NOT AUDITED.",
              "CUT3R cohort identity does not validate camera calibration. No metric scores were recomputed.",
              "This verifies recorded identities, not video hashes or manuscript-number provenance. Compare saved metric_config across policies.",
              "", "## Sources", ""]
    for run in RUNS:
        for metric, results in records[run].items():
            for r in results:
                lines.append(f"- {run} / {metric}: {r['state']} ({len(r['covered'])}/15; {len(r['extra'])} extra) {r['source']}")
                lines.extend(f"  - {issue}" for issue in r["issues"])
    for (run, evaluator), r in suite_checks.items():
        lines.append(f"- {run} / {evaluator}: {r['state']} {r['source']}")
        lines.extend(f"  - {issue}" for issue in r["issues"])
    (args.output / "audit.md").write_text("\n".join(lines) + "\n")
    (args.output / "audit.json").write_text(json.dumps({
        "manifest": str(args.manifest), "expected": sorted(expected), "standard": records,
        "suites": [{"run": run, "evaluator": ev, **r} for (run, ev), r in suite_checks.items()],
        "warnings": warnings,
    }, indent=2) + "\n")
    print("\n".join(lines[:lines.index("## Sources")]))
    print(f"Saved source-by-source audit: {args.output}")
    print(f"Unreadable/unrecognized results or suite warnings: {len(warnings)} (see audit.json)")


if __name__ == "__main__":
    main()
