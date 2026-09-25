"""Build a CPU-only, cross-system lookup-count table; no latency estimates.

MemCam input is the existing retrieval decomposition CSV. WorldMem input is an
explicitly attributed reported summary, not silently treated as audited raw data.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


IDENTITY = ("row", "scene", "dataset_start_frame", "duration_sec")


def integer(value):
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"Expected a finite integer, got {value!r}")
    return int(number)


def memcam_counts(path, run, duration, fps, expected_videos):
    if duration <= 0 or not math.isfinite(fps) or fps <= 0 or expected_videos < 1:
        raise ValueError("Duration, FPS and expected cohort size must be positive")
    grouped, coverage, seen = defaultdict(list), defaultdict(set), set()
    traced = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {*IDENTITY, "run_name", "section_idx", "target_frame", "candidate_count"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            if row["run_name"] != run or integer(row["duration_sec"]) != duration:
                continue
            identity = tuple(row[k] for k in IDENTITY) + (row.get("seed", ""),)
            section, target = integer(row["section_idx"]), integer(row["target_frame"])
            key = (identity, section, target)
            if key in seen:
                raise ValueError(f"Duplicate query: {key}")
            seen.add(key)
            count = integer(row["candidate_count"])
            if count <= 0 or section < 0 or not 0 <= target / fps <= duration:
                raise ValueError(f"Invalid count, section or timestamp: {key}")
            if row.get("candidate_count_mismatch") and integer(row["candidate_count_mismatch"]):
                raise ValueError(f"Candidate reconstruction mismatch: {key}")
            logged = row.get("traced_candidate_count", "")
            if logged and integer(logged) >= 0:
                if integer(logged) != count:
                    raise ValueError(f"Logged and reconstructed counts differ: {key}")
                traced += 1
            # Half-open windows; include the exact final endpoint in window four.
            window = min(3, int((target / fps) / (duration / 4)))
            grouped[(identity, window)].append(count)
            coverage[identity].add((section, target))
    identities = sorted(coverage)
    if len(identities) != expected_videos:
        raise ValueError(f"Expected {expected_videos} trajectories, found {len(identities)}")
    reference = coverage[identities[0]]
    for identity in identities:
        if coverage[identity] != reference:
            raise ValueError(f"Query coverage differs across trajectories: {identity}")
        if any((identity, w) not in grouped for w in range(4)):
            raise ValueError(f"Missing time window: {identity}")
    sections = sorted({s for s, _ in reference})
    if any(b != a + 1 for a, b in zip(sections, sections[1:])):
        raise ValueError("Section coverage has gaps")
    windows = []
    for w in range(4):
        values = [x for i in identities for x in grouped[(i, w)]]
        windows.append({"start_sec": w * duration / 4, "end_sec": (w + 1) * duration / 4,
                        "candidate_mean": mean(mean(grouped[(i, w)]) for i in identities),
                        "candidate_min": min(values), "candidate_max": max(values),
                        "retrieved_per_query": 1, "sampled_queries": len(values)})
    return {"system": "MemCam", "duration_sec": duration, "videos": len(identities),
            "run": run, "source": str(path.resolve()), "windows": windows,
            "aggregation": "Query mean per trajectory/window, then equal-weight trajectory mean; ranges over sampled queries.",
            "evidence": "Reconstructed eligible counts at sampled logged reads; checked against logged counts where present.",
            "queries": len(seen), "queries_with_logged_count": traced,
            "trajectories": [list(i) for i in identities], "fps": fps,
            "query_definition": "One historical selection per target-frame query; 76 context slots per chunk, not 76 unique frames.",
            "sampling": "Uses all matching rows in the supplied diagnostic CSV, not all rollout reads."}


def validate_summary(summary):
    windows = summary["windows"]
    if len(windows) != 4 or summary["videos"] <= 0:
        raise ValueError("Expected four windows and a positive cohort size")
    for index, row in enumerate(windows):
        fields = ("candidate_mean", "candidate_min", "candidate_max", "retrieved_per_query")
        if not all(math.isfinite(row[f]) for f in fields):
            raise ValueError("Nonfinite summary values")
        if not 0 < row["candidate_min"] <= row["candidate_mean"] <= row["candidate_max"]:
            raise ValueError("Invalid candidate summary range")
        if not 0 < integer(row["retrieved_per_query"]) <= row["candidate_min"]:
            raise ValueError("Invalid retrieved count")
        if row["start_sec"] != index * summary["duration_sec"] / 4 or row["end_sec"] != (index + 1) * summary["duration_sec"] / 4:
            raise ValueError("Windows must partition the stated duration into quarters")


def table_rows(summaries):
    result = []
    for summary in summaries:
        validate_summary(summary)
        first = summary["windows"][0]["candidate_mean"]
        for window in summary["windows"]:
            result.append({"system": summary["system"], "duration_sec": summary["duration_sec"],
                           "videos": summary["videos"], **window,
                           "growth": window["candidate_mean"] / first})
    return result


def attach_latency(rows, path, memcam):
    provenance = json.loads(path.with_name("provenance.json").read_text())
    if (provenance.get("status") != "complete" or provenance.get("system") != "MemCam"
            or provenance.get("benchmark") != "isolated_cpu_retrieval_replay"
            or provenance.get("summary_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError("Need a completed MemCam replay with a matching summary hash")
    expected = {(int(i[0]), i[1], int(i[2])) for i in memcam["trajectories"]}
    actual = {(int(i["row"]), i["scene"], int(i["dataset_start_frame"])) for i in provenance["cohort"]}
    if actual != expected or provenance["parameters"]["baseline_run"] != memcam["run"]:
        raise ValueError("Latency and count cohorts/runs differ")
    if provenance["parameters"]["ours_run"] != "slam_b32_covisibility":
        raise ValueError("Latency Ours run must be slam_b32_covisibility")
    with path.open(newline="") as handle:
        timing_rows = list(csv.DictReader(handle))
    indexed = {}
    for r in timing_rows:
        key = (r["system"], float(r["start_sec"]), float(r["end_sec"]))
        if key in indexed or r["system"] != "MemCam" or int(r["duration_sec"]) != memcam["duration_sec"]:
            raise ValueError("Duplicate or incompatible latency window")
        for policy in ("unbounded", "ours"):
            value = float(r[f"{policy}_query_ms"])
            if not math.isfinite(value) or value <= 0 or int(r[f"{policy}_videos"]) != memcam["videos"]:
                raise ValueError("Invalid timing or latency cohort size")
        indexed[key] = r
    matched = 0
    for row in rows:
        if row["system"] != "MemCam":
            continue
        timing = indexed[(row["system"], row["start_sec"], row["end_sec"])]
        row.update({f"{p}_query_ms": float(timing[f"{p}_query_ms"]) for p in ("unbounded", "ours")})
        matched += 1
    if matched != len(indexed):
        raise ValueError("Unmatched latency windows")
    return provenance


def latex(rows):
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{\textbf{Ours bounds the candidate bank while preserving the retrieval interface.} Unbounded counts are observed window means; Ours ($B=32$) shows the enforced candidate-count upper bound, not a measured mean. Numeric query times are measured by isolated retrieval replay; unmeasured cells remain TBD.}",
             r"\label{tab:lookup-work}", r"\begin{tabular}{llrrrr}", r"\toprule",
             r"& & \multicolumn{2}{c}{Frames considered per query} & \multicolumn{2}{c}{Mean query time (ms)} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"System & Generated time (s) & Unbounded (mean) & Ours ($B=32$) & Unbounded & Ours ($B=32$) \\", r"\midrule"]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["system"], row["duration_sec"])].append(row)
    for index, ((system, duration), windows) in enumerate(groups.items()):
        windows = sorted(windows, key=lambda r: r["start_sec"])
        first, last = windows[0], windows[-1]
        if len({r["retrieved_per_query"] for r in windows}) != 1:
            raise ValueError("The fixed-context caption requires constant retrieved counts")
        if index:
            lines.append(r"\midrule")
        for row in (first, last):
            times = [f"{row[f'{p}_query_ms']:.2f}" if f"{p}_query_ms" in row else "TBD"
                     for p in ("unbounded", "ours")]
            lines.append(f"{system} & {row['start_sec']:g}--{row['end_sec']:g} & "
                         f"{row['candidate_mean']:,.1f} & " + r"$\leq 32$ & " + " & ".join(times) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\begin{minipage}{\linewidth}\footnotesize",
              r"Both policies select one frame per query in MemCam (76 context slots per chunk), and eight in WorldMem. Query time covers candidate scoring and selection, excluding encoding, denoising, and bank updates; comparisons require matched hardware and retrieval settings. WorldMem starts with 600 context frames; its duration excludes this initial history. WorldMem counts are reported summaries pending raw-source audit.",
              r"\end{minipage}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path.home() / "memcam_results/context_180s/unbounded_failure_decomposition_180s/tables/query_decomposition.csv")
    parser.add_argument("--run", default="baseline")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--expected-videos", type=int, default=15)
    parser.add_argument("--worldmem-summary", type=Path, default=Path(__file__).resolve().parent / "configs/worldmem_lookup_reported.json")
    parser.add_argument("--latency-summary", type=Path, help="Completed MemCam replay latency_summary.csv with sibling provenance.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    memcam = memcam_counts(args.input, args.run, args.duration, args.fps, args.expected_videos)
    worldmem = json.loads(args.worldmem_summary.read_text())
    summaries = [memcam, worldmem]
    rows = table_rows(summaries)
    timing_provenance = attach_latency(rows, args.latency_summary, memcam) if args.latency_summary else None
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "lookup_work.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "lookup_work.tex").write_text(latex(rows))
    provenance = {"systems": summaries, "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  "worldmem_summary_sha256": hashlib.sha256(args.worldmem_summary.read_bytes()).hexdigest(),
                  "ours_table_column": {"budget": 32, "candidate_upper_bound": 32,
                                        "evidence": "Enforced budget bound, not an observed policy mean; no Ours query logs loaded."},
                  "query_latency": {"status": "MemCam replay complete; WorldMem pending" if timing_provenance else "pending", "unit": "ms/query",
                                    "replay_provenance": timing_provenance,
                                    "scope": "Candidate scoring and selection; excludes encoding, denoising and bank updates.",
                                    "requirement": "Matched hardware and retrieval settings within each system; no conversion from candidate counts."},
                  "limits": "No generation latency, memory-byte measurements or FOV-operation estimates. Query timings, if supplied, are isolated replays. WorldMem summary is user-reported, not raw-data verified. MemCam count checks cover matching CSV rows, not source-video or trace hashes."}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("SYSTEM    WINDOW(s)    ELIGIBLE/QUERY (RANGE)    READ/QUERY  GROWTH")
    for r in rows:
        print(f"{r['system']:9} {r['start_sec']:3g}-{r['end_sec']:<3g}      "
              f"{r['candidate_mean']:7.1f} ({r['candidate_min']}-{r['candidate_max']})"
              f"       {r['retrieved_per_query']:2}       {r['growth']:.2f}x")
    print(f"MemCam: {memcam['videos']} trajectories; {memcam['queries']} sampled reads; "
          f"{memcam['queries_with_logged_count']} counts checked against logs.")
    print("WorldMem: user-reported summary; 600 initial context frames + 600 generated frames at 10 FPS; raw-source audit pending.")
    print(f"Table and provenance: {args.output}")


if __name__ == "__main__":
    main()
