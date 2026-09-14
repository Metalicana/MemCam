"""Diagnose standard/Long ranking differences from saved results, CPU only."""

import argparse
import ast
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np

from audit_metric_coverage import DIMENSIONS, bench_details, finite, valid_bench_score
from collect_matched_video_metrics import POLICIES, collect

GEO = "slam_b32_covisibility"
CONSISTENCY = DIMENSIONS[:2]
COMPONENTS = ("inclip_score", "clip2clip_score", "mapped_clip2clip_score")


def standard_scores(payload, run, expected):
    scores, cohorts = {}, {}
    for dim in DIMENSIONS:
        scale = 100. if dim == "imaging_quality" else 1.
        details = bench_details(payload.get(dim))
        if details is None:
            raise ValueError(f"Missing dimension: {dim}")
        names = []
        for row in details:
            path = Path(row["video_path"])
            if path.parent.name != run:
                raise ValueError(f"Other policy in {dim}: {path}")
            if path.name in names:
                raise ValueError(f"Duplicate {dim}/{path.name}")
            if not valid_bench_score(dim, row.get("video_results")):
                raise ValueError(f"Invalid score: {dim}/{path.name}")
            names.append(path.name)
            if path.name in expected:
                scores.setdefault(path.name, {})[dim] = float(row["video_results"]) / scale
        cohorts[dim] = {"total": len(names), "matched": len(expected & set(names)),
                        "missing": sorted(expected - set(names)), "extra": sorted(set(names) - expected),
                        "stored_aggregate": payload[dim][0],
                        "all_video_mean": mean(float(r["video_results"]) / scale for r in details) if details else None,
                        "matched_mean": mean(scores[n][dim] for n in expected & set(names)) if expected & set(names) else None}
    return scores, cohorts


def paired(left, right, names, key, draws=10000):
    selected = sorted(n for n in names if key in left.get(n, {}) and key in right.get(n, {}))
    if not selected:
        return None
    differences = np.array([left[n][key] - right[n][key] for n in selected])
    rng = np.random.default_rng(20260914)
    samples = differences[rng.integers(len(selected), size=(draws, len(selected)))].mean(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return {"n": len(selected), "mean_delta": float(differences.mean()),
            "ci_low": float(low), "ci_high": float(high),
            "wins": int((differences > 1e-12).sum()), "losses": int((differences < -1e-12).sum()),
            "per_video": [{"video": n, "delta": float(d)} for n, d in zip(selected, differences)]}


def recorded_args(log):
    if not log.exists():
        return {}
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("args: Namespace("):
            call = ast.parse(line[len("args: "):], mode="eval").body
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "Namespace":
                raise ValueError("Invalid logged Namespace")
            return {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    return {}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnose(root, manifest, draws):
    items = {i: json.loads(line) for i, line in enumerate(manifest.read_text().splitlines()) if line.strip()}
    cohort = {i: r for i, r in items.items() if int(r["duration_sec"]) == 60}
    expected = {r["output_prefix"] + "custom.mp4" for r in cohort.values()}
    if len(cohort) != 15 or len(expected) != 15:
        raise ValueError("Expected exactly 15 unique canonical videos")
    standard, long = {}, {}
    raw, inventory, provenance, issues = {}, {}, [], []
    for run in POLICIES:
        files = sorted((root / "vbench_results" / run).glob("*_eval_results.json"))
        if not files:
            issues.append(f"{run}: no standard VBench result")
            continue
        # Exactly mirror the historical summarizer's latest-file selection.
        path = files[-1]
        payload = json.loads(path.read_text())
        raw[str(path)] = payload
        try:
            standard[run], inventory[run] = standard_scores(payload, run, expected)
            if any(c["matched"] != 15 for c in inventory[run].values()):
                issues.append(f"{run}: latest standard file lacks some canonical scores; do not infer a 15-video reversal")
        except (ValueError, KeyError, TypeError) as exc:
            issues.append(f"{path}: {exc}")
        provenance.append({"run": run, "evaluator": "standard", "source": str(path),
                           "other_files_not_selected": list(map(str, files[:-1])),
                           "historical_video_hashes": "not recorded by the standard launcher"})

    suites = collect(root / "matched15_metrics_60s")
    issues.extend(suites["warnings"])
    acceptable = set()
    for record in suites["coverage"]:
        if record["evaluator"] != "vbench-long":
            continue
        issues.extend(f"{record['run']}: {v}" for v in record["issues"])
        if "suite" not in record:
            continue
        suite = Path(record["suite"])
        saved = {i: json.loads(line) for i, line in enumerate((suite / "manifest.jsonl").read_text().splitlines()) if line.strip()}
        if {i: r for i, r in saved.items() if int(r["duration_sec"]) == 60} != cohort:
            issues.append(f"{record['run']}: Long canonical manifest mismatch")
            continue
        acceptable.add(record["run"])
        freeze = suite / "pip_freeze.txt"
        provenance.append({"run": record["run"], "evaluator": "long", "suite": str(suite),
                           "pip_freeze": sorted(freeze.read_text().splitlines()) if freeze.exists() else None})
    for row in suites["per_video"]:
        if row["evaluator"] != "vbench-long" or row["run"] not in acceptable:
            continue
        path = Path(row["source_json"])
        payload = json.loads(path.read_text())
        raw[str(path)] = payload
        scores = dict(row["scores"])
        for dim in DIMENSIONS:
            details = bench_details(payload[dim])
            detail = details[0]
            scale = 100. if dim == "imaging_quality" else 1.
            if not np.isclose(float(detail["video_results"]) / scale, scores[dim], atol=1e-7, rtol=1e-6):
                issues.append(f"{row['run']}/{row['video']}/{dim}: top-level/detail scale mismatch")
            if dim in CONSISTENCY:
                for component in COMPONENTS:
                    if finite(detail.get(component)):
                        scores[f"{dim}.{component}"] = detail[component]
                    else:
                        issues.append(f"{row['run']}/{row['video']}/{dim}: missing {component}")
                if all(finite(detail.get(c)) for c in COMPONENTS):
                    scores[f"{dim}.residual_from_half_blend"] = scores[dim] - .5 * (detail["inclip_score"] + detail["mapped_clip2clip_score"])
                    if detail["clip2clip_score"] == 0:
                        issues.append(f"{row['run']}/{row['video']}/{dim}: zero cross-clip score; inspect grouping")
        long.setdefault(row["run"], {})[row["video"]] = scores
        smoke = path.parent.parent
        spec = json.loads((smoke / "spec.json").read_text())
        source = Path(spec["source_video"])
        unchanged = (source.is_file() and source.stat().st_size == spec.get("source_bytes")
                     and source.stat().st_mtime_ns == spec.get("source_mtime_ns"))
        if not unchanged:
            issues.append(f"{row['run']}/{row['video']}: current source size/mtime differs or file unavailable")
        provenance.append({"run": row["run"], "video": row["video"], "evaluator": "long_row",
                           "spec": spec, "source_size_mtime_unchanged": unchanged,
                           "logged_arguments": recorded_args(smoke / "vbench-long.log")})
    # Compare recorded settings, excluding paths that necessarily differ per row.
    setting_keys = ("dev_flag", "sb_clip2clip_feat_extractor", "bg_clip2clip_feat_extractor",
                    "read_frame", "use_semantic_splitting", "static_filter_flag", "imaging_quality_preprocessing_mode",
                    "clip_length_config", "slow_fast_eval_config", "subject_mapping_file_path", "background_mapping_file_path")
    settings = {json.dumps({k: p["logged_arguments"].get(k) for k in setting_keys}, sort_keys=True)
                for p in provenance if p["evaluator"] == "long_row" and p["logged_arguments"]}
    if len(settings) > 1:
        issues.append("Long invocation settings differ across rows/policies; inspect provenance")
    freezes = {json.dumps(p["pip_freeze"]) for p in provenance if p["evaluator"] == "long" and p["pip_freeze"] is not None}
    if len(freezes) > 1:
        issues.append("Long pip-freeze differs across suites; inspect package changes")
    adapters = {p["spec"].get("long_grouping_adapter_sha256") for p in provenance if p["evaluator"] == "long_row"}
    if len(adapters) > 1:
        issues.append("Long grouping adapter hashes differ across rows/policies")
    comparisons = []
    for opponent in ("baseline", "fifo_b32", "ri_b32_dino_rgb"):
        common = expected.intersection(standard.get(GEO, {}), standard.get(opponent, {}),
                                       long.get(GEO, {}), long.get(opponent, {}))
        for dim in DIMENSIONS:
            for family, data, key in [("standard", standard, dim), ("long", long, dim)] + [
                    (component, long, f"{dim}.{component}") for component in COMPONENTS if dim in CONSISTENCY]:
                stats = paired(data.get(GEO, {}), data.get(opponent, {}), common, key, draws)
                if stats:
                    comparisons.append({"opponent": opponent, "dimension": dim, "component": family, **stats})
    return {"expected": sorted(expected), "standard": standard, "long": long, "inventory": inventory,
            "comparisons": comparisons, "provenance": provenance, "issues": issues, "raw_results": raw}


def report_text(data):
    lines = ["# VBench ranking diagnosis", "", "## Cohort effect on standard VBench", "",
             "| Run | Metric | File N | Matched N | Stored aggregate | Matched mean |", "|---|---|---:|---:|---:|---:|"]
    for run, dimensions in data["inventory"].items():
        for dim, c in dimensions.items():
            subset = f"{c['matched_mean']:.6f}" if c["matched"] == 15 else "incomplete"
            lines.append(f"| {run} | {dim} | {c['total']} | {c['matched']} | {c['stored_aggregate']:.6f} | {subset} |")
    lines += ["", "## GeoCov minus comparator on identical videos", "",
              "Positive favors GeoCov. Comparisons use the same intersection in BOTH evaluators.",
              "RI's incomplete intersection is exploratory, not the primary 15-video result.", "",
              "| Comparator | Metric | Component | N | Delta | 95% paired bootstrap CI | Wins/losses |",
              "|---|---|---|---:|---:|---|---|"]
    for c in data["comparisons"]:
        lines.append(f"| {c['opponent']} | {c['dimension']} | {c['component']} | {c['n']} | {c['mean_delta']:+.6f} | "
                     f"[{c['ci_low']:+.6f}, {c['ci_high']:+.6f}] | {c['wins']}/{c['losses']} |")
    lines += ["", "## Fusion checks", "",
              "Cross-clip range -> mapped range; residual against a 50/50 blend is diagnostic, not an assumed runtime configuration."]
    for run, videos in data["long"].items():
        for dim in CONSISTENCY:
            raw = [s[f"{dim}.clip2clip_score"] for s in videos.values() if f"{dim}.clip2clip_score" in s]
            mapped = [s[f"{dim}.mapped_clip2clip_score"] for s in videos.values() if f"{dim}.mapped_clip2clip_score" in s]
            residual = [abs(s[f"{dim}.residual_from_half_blend"]) for s in videos.values() if f"{dim}.residual_from_half_blend" in s]
            if raw and mapped and residual:
                lines.append(f"- {run}/{dim}: [{min(raw):.6f}, {max(raw):.6f}] -> "
                             f"[{min(mapped):.6f}, {max(mapped):.6f}]; max residual {max(residual):.8f}")
    lines += ["", "## Interpretation", ""]
    for opponent in ("baseline", "fifo_b32"):
        for dim in DIMENSIONS:
            pair = {c["component"]: c for c in data["comparisons"] if c["opponent"] == opponent and c["dimension"] == dim}
            if not all(k in pair and pair[k]["n"] == 15 for k in ("standard", "long")):
                lines.append(f"- {opponent}/{dim}: full matched comparison unavailable.")
                continue
            a, b = pair["standard"]["mean_delta"], pair["long"]["mean_delta"]
            verdict = "point-estimate ranking reverses after cohort matching" if a * b < 0 else "no sign reversal after cohort matching"
            lines.append(f"- {opponent}/{dim}: {verdict} (standard {a:+.6f}, Long {b:+.6f}).")
    lines += ["", "These are exploratory trajectory-level bootstrap intervals, unadjusted for multiple comparisons.",
              "Raw cross-clip and mapped scores use different scales. Mapping/clip splitting can change rankings without a software bug.",
              "Imaging-quality per-video scores are divided by 100 in BOTH evaluators to match their published aggregate scale.",
              "No scores were regenerated. A genuine tradeoff remains possible; this report does not force agreement.",
              "Historical standard-VBench video hashes/environments and frozen upstream Long configs were not recorded by our launchers.",
              "Matching names/settings alone cannot establish unchanged input bytes or identical historical evaluator code.",
              "", "## Checks / missing evidence", ""]
    lines.extend("- " + issue for issue in data["issues"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "memcam_results")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("Use at least 100 bootstrap draws")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    data = diagnose(args.root, args.manifest, args.bootstrap)
    args.output.mkdir(parents=True, exist_ok=True)
    text = report_text(data)
    (args.output / "diagnosis.md").write_text(text)
    (args.output / "diagnosis.json").write_text(json.dumps(data, indent=2) + "\n")
    rows = [{"opponent": c["opponent"], "dimension": c["dimension"], "component": c["component"], **r}
            for c in data["comparisons"] for r in c["per_video"]]
    write_csv(args.output / "paired_video_differences.csv", rows)
    print(text)
    print(f"Saved diagnosis and raw result records: {args.output}")


if __name__ == "__main__":
    main()
