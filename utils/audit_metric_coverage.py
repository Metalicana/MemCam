"""Inventory generated videos and metric coverage without loading GPU libraries.

This is an output inventory, not a decode/provenance audit or an automatic
permission to skip evaluation. Exact original-video names are required for
VBench-Long; split-clip results alone do not establish full-video coverage.
"""

import argparse
import json
import math
from pathlib import Path
import re
import sys


DIMENSIONS = (
    "subject_consistency", "background_consistency", "motion_smoothness",
    "dynamic_degree", "aesthetic_quality", "imaging_quality",
)
VIDEO_NAME = re.compile(r"_(\d+)s_custom\.mp4$")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path, warnings):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        warnings.append(f"Cannot read {path}: {exc}")
        return {}


def read_rows(path, warnings):
    if not path.exists():
        return []
    rows = []
    try:
        with path.open() as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
                except ValueError:
                    warnings.append(f"Invalid JSON at {path}:{number}")
    except OSError as exc:
        warnings.append(f"Cannot read {path}: {exc}")
    return rows


def video_key(value):
    path = Path(str(value or ""))
    return path.parent.name, path.name


def inventory(root, duration=None):
    warnings = []
    groups = {}
    index = {}
    candidates = set(root.rglob("*_custom.mp4"))
    for status_path in root.rglob("run_status.jsonl"):
        for row in read_rows(status_path, warnings):
            name = Path(str(row.get("output", ""))).name
            if VIDEO_NAME.search(name):
                candidates.add(status_path.parent / name)
    for path in sorted(candidates):
        match = VIDEO_NAME.search(path.name)
        if not match or any(part in {"split_clip", "split_scene"} for part in path.parts):
            continue
        seconds = int(match[1])
        if duration is not None and seconds != duration:
            continue
        group = groups.setdefault((path.parent, seconds), {
            "directory": str(path.parent), "run": path.parent.name,
            "duration": seconds, "present": set(), "logged_done": set(),
            "lpips": set(), "fvd_groups": [],
            "vbench": {d: set() for d in DIMENSIONS},
            "vbench_long": {d: set() for d in DIMENSIONS}, "sources": set(),
        })
        if path.exists() and path.stat().st_size > 0:
            group["present"].add(path.name)
        index.setdefault(video_key(path), []).append(group)

    # Names repeat across machines, so permit relocated paths but reject ambiguous
    # run/name identities instead of assigning one result to multiple runs.
    def lookup(value):
        matches = index.get(video_key(value), [])
        return matches[0] if len(matches) == 1 else None

    for key, matches in index.items():
        if len(matches) > 1:
            warnings.append(f"Ambiguous run/video identity: {key}; metric matching disabled")
    for directory in {key[0] for key in groups}:
        state = {}
        for row in read_rows(directory / "run_status.jsonl", warnings):
            name = Path(str(row.get("output", ""))).name
            if row.get("status") != "skipped":
                state[name] = row.get("status")
        for (parent, _), group in groups.items():
            if parent == directory:
                group["logged_done"] = {
                    name for name in group["present"] if state.get(name) == "completed"
                }

    for path in sorted(root.rglob("metrics.jsonl")):
        rows = read_rows(path, warnings)
        for row in rows:
            group = lookup(row.get("output"))
            if group is None or str(row.get("duration_sec")) != str(group["duration"]):
                continue
            name = Path(row["output"]).name
            if name not in group["present"]:
                continue
            if row.get("status") == "completed" and finite(row.get("lpips_alex")):
                group["lpips"].add(name)
                group["sources"].add(str(path))

        summary_path = path.with_name("summary.json")
        if not summary_path.exists():
            continue
        summary = read_json(summary_path, warnings)
        if not isinstance(summary, dict):
            continue
        for seconds, result in summary.get("by_duration", {}).items():
            if not isinstance(result, dict) or not finite(result.get("fvd")):
                continue
            members = {}
            for row in rows:
                group = lookup(row.get("output"))
                if (group is None or str(group["duration"]) != seconds
                        or str(row.get("duration_sec")) != seconds
                        or row.get("status") not in {"completed", "short_video"}):
                    continue
                members.setdefault((group["directory"], group["duration"]), []).append(row)
            for (directory, seconds_int), selected in members.items():
                group = groups[(Path(directory), seconds_int)]
                names = {Path(row["output"]).name for row in selected}
                group["fvd_groups"].append({
                    "source": str(summary_path), "fvd": result["fvd"],
                    "videos": sorted(names),
                    "short_videos": sum(r["status"] == "short_video" for r in selected),
                    "matches_present_set": names == group["present"],
                    "reported_count": result.get("completed_or_short"),
                    "metric_config": summary.get("metric_config", {}),
                })

    for path in sorted(root.rglob("*_eval_results.json")):
        family = "vbench_long" if "long" in str(path.parent).lower() else "vbench"
        payload = read_json(path, warnings)
        if not isinstance(payload, dict):
            continue
        unmatched = 0
        for dim in DIMENSIONS:
            value = payload.get(dim)
            if not isinstance(value, list) or len(value) != 2 or not isinstance(value[1], list):
                continue
            for detail in value[1]:
                if not isinstance(detail, dict):
                    continue
                group = lookup(detail.get("video_path"))
                score = detail.get("video_results")
                if group is None:
                    unmatched += 1
                    continue
                name = Path(str(detail.get("video_path"))).name
                if name in group["present"] and finite(score):
                    group[family][dim].add(name)
                    group["sources"].add(str(path))
        if unmatched:
            warnings.append(f"{path}: {unmatched} unmatched dimension/video records (possibly split clips or other durations)")

    report = []
    for _, group in sorted(groups.items()):
        for family in ("vbench", "vbench_long"):
            covered = set.intersection(*group[family].values())
            group[family + "_complete"] = sorted(covered)
            group[family + "_missing"] = {
                dim: sorted(group["present"] - group[family][dim]) for dim in DIMENSIONS
            }
            group[family] = {dim: sorted(names) for dim, names in group[family].items()}
        group["lpips_missing"] = sorted(group["present"] - group["lpips"])
        for key in ("present", "logged_done", "lpips", "sources"):
            group[key] = sorted(group[key])
        report.append(group)
    return {"runs": report, "warnings": warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "memcam_results")
    parser.add_argument("--duration", type=int, help="Full source-video duration, e.g. 60 or 180")
    parser.add_argument("--expected", type=int, help="Expected videos per run/duration (optional)")
    parser.add_argument("--json", type=Path, help="Write detailed coverage and missing-video lists")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"Results root does not exist: {args.root}")
    report = inventory(args.root, args.duration)
    print(f"{'RUN':42} {'SEC':>4} {'FILES':>7} {'DONE*':>6} {'LPIPS':>6} {'FVD N**':>8} {'VB6':>5} {'VBL6':>5}")
    for group in report["runs"]:
        n = len(group["present"])
        files = f"{n}/{args.expected}" if args.expected is not None else str(n)
        fvd = ",".join(str(n) for n in sorted({len(g['videos']) for g in group['fvd_groups']})) or "--"
        print(f"{group['run']:42} {group['duration']:4} {files:>7} {len(group['logged_done']):6} "
              f"{len(group['lpips']):6} {fvd:>8} {len(group['vbench_complete']):5} {len(group['vbench_long_complete']):5}")
        print(f"  {group['directory']}")
    print("\nDONE*: nonempty files with a completion log; not decoded/integrity-checked.")
    print("LPIPS: full-duration, completed rows only. FVD N**: group sizes, not per-video FVD.")
    print("VB6/VBL6: original videos with finite scores in ALL six dimensions (union of result files).")
    print("Inventory only: verify configs/provenance before skipping jobs; split-clip names are not counted as originals.")
    if not report["runs"]:
        print("No matching videos or generation logs found under this root.")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Detailed report: {args.json}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
