"""Collect the matched-15 VBench-Long/CUT3R batch into Markdown, CSV and JSON.

CPU only. Uses the latest non-dry-run attempt per policy/evaluator, never mixes
retries, and only summarizes complete cohorts with matching manifest entries.
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, stdev

from audit_metric_coverage import DIMENSIONS, finite
from smoke_video_metrics import load, validate_long


POLICIES = {
    "baseline": "Complete retention",
    "fifo_b32": "FIFO-32",
    "ri_b32_dino_rgb": "RI-32",
    "slam_b32_covisibility": "GeoCov-32",
}
EVALUATORS = ("vbench-long", "cut3r")
CUT_METADATA = {"run_name", "row", "scene", "start_frame", "duration_sec",
                "frame_stride", "total_video_frames", "reconstruction_dir"}


def read_video(work, run, evaluator, row, item):
    attempts = list((work / f"row_{row:03d}").glob("smoke_*/spec.json"))
    if len(attempts) != 1:
        raise ValueError(f"Expected one row attempt, found {len(attempts)}")
    directory = attempts[0].parent
    spec = load(attempts[0])
    name = item["output_prefix"] + "custom.mp4"
    if (spec.get("dry_run") or spec.get("row") != row
            or spec.get("manifest_item") != item
            or Path(spec["source_video"]).name != name
            or Path(spec["source_video"]).parent.name != run):
        raise ValueError("Source identity or manifest mismatch")
    status = load(directory / "status.json")
    if status.get(evaluator, {}).get("status") != "passed":
        raise ValueError(f"Evaluator did not pass: {status.get(evaluator)}")
    if evaluator == "vbench-long":
        output = directory / "vbench_long_results"
        paths = list(output.glob("*_eval_results.json"))
        if len(paths) != 1:
            raise ValueError("Expected exactly one VBench-Long result file")
        validate_long(output, name)
        data = load(paths[0])
        # Top-level scores include upstream imaging-quality /100 normalization.
        scores = {dimension: data[dimension][0] for dimension in DIMENSIONS}
        source = paths[0]
    else:
        output = directory / "cut3r_metrics"
        failures = load(output / "cut3r_camera_failures.json")
        source = output / "cut3r_camera_metrics.json"
        results = load(source)
        if failures or len(results) != 1:
            raise ValueError("CUT3R scoring failed or is incomplete")
        data = results[0]
        if (data.get("run_name") != run or data.get("row") != row
                or data.get("duration_sec") != 60
                or data.get("scene") != item["scene"]
                or data.get("start_frame") != item["start_frame"]):
            raise ValueError("CUT3R result identity mismatch")
        scores = {key: value for key, value in data.items() if key not in CUT_METADATA}
        for key in ("rotation_error_deg_mean", "translation_error_scale_only_mean",
                    "translation_error_sim3_mean"):
            if not finite(scores.get(key)):
                raise ValueError(f"Invalid required CUT3R score: {key}")
    return {"run": run, "evaluator": evaluator, "row": row, "video": name,
            "source_json": str(source), "scores": scores}


def collect(root):
    if not root.is_dir():
        raise ValueError(f"Results directory does not exist: {root}")
    candidates = {}
    warnings = []
    for path in root.glob("*/suite_status.json"):
        try:
            status = load(path)
            key = (status["run"], status["evaluator"])
            if status.get("dry_run") or key[0] not in POLICIES or key[1] not in EVALUATORS:
                continue
            candidates.setdefault(key, []).append(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"Cannot read {path}: {exc}")
    coverage, videos, summaries = [], [], []
    reference = None
    for run in POLICIES:
        for evaluator in EVALUATORS:
            paths = sorted(candidates.get((run, evaluator), []),
                           key=lambda p: (p.stat().st_mtime_ns, str(p)))
            record = {"run": run, "evaluator": evaluator, "valid": 0, "expected": 15,
                      "complete": False, "issues": []}
            coverage.append(record)
            if not paths:
                record["issues"].append("No non-dry-run suite status found")
                continue
            path = paths[-1]
            record["suite"] = str(path.parent)
            if len(paths) > 1:
                warnings.append(f"{run}/{evaluator}: selected newest of {len(paths)} attempts: {path.parent}")
            try:
                status = load(path)
                items = {i: json.loads(line) for i, line in enumerate(
                    (path.parent / "manifest.jsonl").read_text().splitlines()) if line.strip()}
                cohort = {i: item for i, item in items.items() if int(item["duration_sec"]) == 60}
                if len(cohort) != 15 or status["rows"] != list(cohort):
                    raise ValueError("Suite does not match exactly 15 original 60s manifest rows")
                if reference is None:
                    reference = cohort
                if cohort != reference:
                    raise ValueError("Cohort differs from the other policy/evaluator suites")
                rows = status["results"]
                by_row = {row["row"]: row for row in rows}
                if len(by_row) != len(rows) or set(by_row) - set(cohort):
                    raise ValueError("Duplicate or unexpected rows in suite status")
                valid = []
                for row, item in cohort.items():
                    try:
                        if by_row.get(row, {}).get("exit_code") != 0:
                            raise ValueError("Row failed or did not finish")
                        valid.append(read_video(path.parent, run, evaluator, row, item))
                    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                        record["issues"].append(f"Row {row}: {exc}")
                videos.extend(valid)
                record["valid"] = len(valid)
                record["complete"] = len(valid) == 15
                if record["complete"]:
                    for metric in sorted({key for v in valid for key in v["scores"]}):
                        values = [v["scores"].get(metric) for v in valid]
                        values = [value for value in values if finite(value)]
                        summaries.append({"run": run, "evaluator": evaluator, "metric": metric,
                                          "n": len(values), "mean": mean(values) if values else None,
                                          "std": stdev(values) if len(values) > 1 else None,
                                          "median": median(values) if values else None,
                                          "min": min(values) if values else None,
                                          "max": max(values) if values else None})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record["issues"].append(str(exc))
    return {"coverage": coverage, "per_video": videos, "summary": summaries, "warnings": warnings}


def markdown(report):
    lines = ["# MemCam: Matched 15 Videos, 60 Seconds", "",
             "VBench-Long and CUT3R only. This job did not compute LPIPS, FVD or standard VBench.",
             "CUT3R is provisional: ground-truth camera calibration remains unresolved.", "",
             "## Coverage", "", "| Policy | VBench-Long | CUT3R |", "|---|---:|---:|"]
    coverage = {(r["run"], r["evaluator"]): r for r in report["coverage"]}
    for run, name in POLICIES.items():
        lines.append("| " + name + " | " + " | ".join(
            f"{coverage[run, ev]['valid']}/15" for ev in EVALUATORS) + " |")
    lookup = {(r["run"], r["evaluator"], r["metric"]): r for r in report["summary"]}
    for evaluator, heading in (("vbench-long", "VBench-Long (Higher Is Better)"),
                               ("cut3r", "CUT3R (Provisional)")):
        lines += ["", "## " + heading, "",
                  "| Metric | " + " | ".join(POLICIES.values()) + " |",
                  "|---|" + "---:|" * len(POLICIES)]
        metrics = list(DIMENSIONS) if evaluator == "vbench-long" else sorted(
            {r["metric"] for r in report["summary"] if r["evaluator"] == evaluator})
        for metric in metrics:
            cells = []
            for run in POLICIES:
                score = lookup.get((run, evaluator, metric))
                value = f"{score['mean']:.6f}" if score and score["mean"] is not None else "--"
                if score and score["n"] != 15:
                    value += f" (n={score['n']})"
                cells.append(value)
            lines.append("| " + metric + " | " + " | ".join(cells) + " |")
    lines += ["", "Values are equal-video means, not pooled-frame or pooled-clip means.",
              "VBench-Long imaging quality uses the evaluator's normalized scale.",
              "CUT3R error metrics: lower is better; camera-control score: higher is better.",
              "Path lengths, scales, ratios and sampled-frame counts are diagnostics, not ranked quality metrics.",
              "No aggregate is shown for an incomplete or mismatched cohort.",
              "CSV summaries also include sample standard deviation, median, minimum, maximum and n."]
    issues = report["warnings"] + [f"{r['run']}/{r['evaluator']}: {issue}"
                                   for r in report["coverage"] for issue in r["issues"]]
    if issues:
        lines += ["", "## Issues", ""] + ["- " + issue for issue in issues]
    lines += ["", "## Selected Suites", ""] + [
        f"- {r['run']}/{r['evaluator']}: {r['suite']}" for r in report["coverage"] if "suite" in r]
    return "\n".join(lines) + "\n"


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "memcam_results/matched15_metrics_60s")
    parser.add_argument("--output", type=Path, help="Default: <root>/report")
    args = parser.parse_args()
    report = collect(args.root)
    output = args.output or args.root / "report"
    output.mkdir(parents=True, exist_ok=True)
    text = markdown(report)
    (output / "summary.md").write_text(text)
    (output / "all_results.json").write_text(json.dumps(report, indent=2) + "\n")
    for evaluator in EVALUATORS:
        summary = [row for row in report["summary"] if row["evaluator"] == evaluator]
        write_csv(output / f"{evaluator}_summary.csv", summary,
                  ["run", "evaluator", "metric", "n", "mean", "std", "median", "min", "max"])
        raw = [{**{k: v for k, v in row.items() if k != "scores"}, "metric": key, "value": value}
               for row in report["per_video"] if row["evaluator"] == evaluator
               for key, value in row["scores"].items()]
        write_csv(output / f"{evaluator}_per_video.csv", raw,
                  ["run", "evaluator", "row", "video", "source_json", "metric", "value"])
    print(text)
    print(f"Saved reports: {output}")
    raise SystemExit(0 if all(r["complete"] for r in report["coverage"]) else 1)


if __name__ == "__main__":
    main()
