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


def latex(rows):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Eligible archive size at sampled unbounded retrieval queries. Windows partition each system's own rollout; horizons and query definitions differ. Means and observed ranges are shown; growth is relative to the first window within each system. These are candidate counts, not measured latency. WorldMem values are reported summaries pending source audit.}",
             r"\label{tab:lookup-work}", r"\begin{tabular}{llrrr}", r"\toprule",
             r"System & Window (s) & Eligible/query & Read/query & Growth \\", r"\midrule"]
    previous = None
    for row in rows:
        system = row["system"]
        if system != previous and previous is not None:
            lines.append(r"\midrule")
        label = system if system != previous else ""
        lines.append(f"{label} & {row['start_sec']:g}--{row['end_sec']:g} & "
                     f"{row['candidate_mean']:,.1f} ({row['candidate_min']:,.0f}--{row['candidate_max']:,.0f}) & "
                     f"{row['retrieved_per_query']} & {row['growth']:.2f}" + r"$\times$ \\")
        previous = system
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\begin{minipage}{\linewidth}\footnotesize",
              r"MemCam selects one memory per target-frame query to populate 76 context slots per chunk; repeated indices are allowed. WorldMem reports eight memories per read. Each cohort has its separately reported size in the accompanying CSV. WorldMem's first window starts with 600 eligible memories; the initial-history protocol requires confirmation. MemCam means weight trajectories equally; WorldMem weighting awaits verification. Absolute counts are not a cross-system speed comparison.",
              r"\end{minipage}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path.home() / "memcam_results/context_180s/unbounded_failure_decomposition_180s/tables/query_decomposition.csv")
    parser.add_argument("--run", default="baseline")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--expected-videos", type=int, default=15)
    parser.add_argument("--worldmem-summary", type=Path, default=Path(__file__).with_name("worldmem_lookup_reported.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    memcam = memcam_counts(args.input, args.run, args.duration, args.fps, args.expected_videos)
    worldmem = json.loads(args.worldmem_summary.read_text())
    summaries = [memcam, worldmem]
    rows = table_rows(summaries)
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
                  "limits": "No wall-clock timings, memory-byte measurements or FOV-operation estimates. WorldMem summary is user-reported, not raw-data verified. MemCam completeness checks cover matching CSV rows, not source-video or trace hashes."}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("SYSTEM    WINDOW(s)    ELIGIBLE/QUERY (RANGE)    READ/QUERY  GROWTH")
    for r in rows:
        print(f"{r['system']:9} {r['start_sec']:3g}-{r['end_sec']:<3g}      "
              f"{r['candidate_mean']:7.1f} ({r['candidate_min']}-{r['candidate_max']})"
              f"       {r['retrieved_per_query']:2}       {r['growth']:.2f}x")
    print(f"MemCam: {memcam['videos']} trajectories; {memcam['queries']} sampled reads; "
          f"{memcam['queries_with_logged_count']} counts checked against logs.")
    print("WorldMem: user-reported summary; initial history and weighting pending audit.")
    print(f"Table and provenance: {args.output}")


if __name__ == "__main__":
    main()
