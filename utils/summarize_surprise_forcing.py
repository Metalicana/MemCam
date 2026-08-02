import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def load_trace(path):
    admissions = []
    routes = []
    usages = []
    bank_updates = []
    metadata = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("row", "scene", "duration_sec", "run_name"):
                if row.get(key) is not None:
                    metadata[key] = row[key]
            event = row.get("event")
            if event == "surprise_admission":
                admissions.append(row)
            elif event == "surprise_routing":
                routes.append(row)
            elif event == "surprise_usage":
                usages.append(row)
            elif event == "surprise_bank_update":
                bank_updates.append(row)
    return metadata, admissions, routes, usages, bank_updates


def summarize_trace(path):
    metadata, admissions, routes, usages, bank_updates = load_trace(path)
    rejection_counts = Counter(
        row.get("rejection_reason") or "none" for row in admissions
    )
    non_warmup = [row for row in admissions if not row.get("warmup", False)]
    final_update = bank_updates[-1] if bank_updates else {}
    routed_counts = [len(row.get("routed_memory_frames", [])) for row in routes]
    retrieved_counts = [
        int(row.get("actually_retrieved_count", 0)) for row in usages
    ]
    return {
        "source_file": path.name,
        "row": metadata.get("row"),
        "scene": metadata.get("scene"),
        "duration_sec": metadata.get("duration_sec"),
        "evaluated": len(admissions),
        "warmup_evaluated": len(admissions) - len(non_warmup),
        "gate_passes": sum(bool(row.get("gate_pass")) for row in admissions),
        "non_warmup_gate_passes": sum(
            bool(row.get("gate_pass")) for row in non_warmup
        ),
        "commits": sum(bool(row.get("committed")) for row in admissions),
        "replacements": sum(
            row.get("evicted_frame") is not None for row in admissions
        ),
        "gate_rejections": rejection_counts["surprise_gate"],
        "priority_rejections": rejection_counts["priority"],
        "surprise_mean": mean(row.get("surprise") for row in admissions),
        "prediction_surprise_mean": mean(
            row.get("prediction_surprise") for row in admissions
        ),
        "novelty_surprise_mean": mean(
            row.get("novelty_surprise") for row in admissions
        ),
        "route_events": len(routes),
        "routed_frames_mean": mean(routed_counts),
        "actually_retrieved_frames_mean": mean(retrieved_counts),
        "final_bank_size": len(final_update.get("bank_frames", [])),
        "final_threshold": final_update.get("threshold"),
        "final_ema_mean": final_update.get("ema_mean"),
        "final_ema_variance": final_update.get("ema_variance"),
    }


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def build_summary(rows, expected_videos=None):
    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "evaluated",
            "warmup_evaluated",
            "gate_passes",
            "non_warmup_gate_passes",
            "commits",
            "replacements",
            "gate_rejections",
            "priority_rejections",
            "route_events",
        )
    }
    non_warmup_evaluated = totals["evaluated"] - totals["warmup_evaluated"]
    return {
        "scope": {
            "trace_files": len(rows),
            "expected_videos": expected_videos,
            "scenes": sorted(
                {row["scene"] for row in rows if row.get("scene") is not None}
            ),
        },
        "controller": {
            **totals,
            "gate_pass_rate": ratio(totals["gate_passes"], totals["evaluated"]),
            "non_warmup_gate_pass_rate": ratio(
                totals["non_warmup_gate_passes"], non_warmup_evaluated
            ),
            "commit_rate": ratio(totals["commits"], totals["evaluated"]),
            "replacement_rate_per_commit": ratio(
                totals["replacements"], totals["commits"]
            ),
            "surprise_mean": mean(row["surprise_mean"] for row in rows),
            "prediction_surprise_mean": mean(
                row["prediction_surprise_mean"] for row in rows
            ),
            "novelty_surprise_mean": mean(
                row["novelty_surprise_mean"] for row in rows
            ),
            "final_threshold_mean": mean(row["final_threshold"] for row in rows),
            "final_bank_size_mean": mean(row["final_bank_size"] for row in rows),
            "routed_frames_mean": mean(row["routed_frames_mean"] for row in rows),
            "actually_retrieved_frames_mean": mean(
                row["actually_retrieved_frames_mean"] for row in rows
            ),
        },
        "per_video": rows,
    }


def format_number(value):
    return "n/a" if value is None else f"{float(value):.4f}"


def write_report(path, summary):
    scope = summary["scope"]
    controller = summary["controller"]
    expected = scope["expected_videos"]
    expected_text = "" if expected is None else f" / {expected} expected"
    lines = [
        "# Surprise Forcing Memory Summary",
        "",
        f"- Trace files: {scope['trace_files']}{expected_text}",
        f"- Candidate frames evaluated: {controller['evaluated']}",
        f"- Gate pass rate: {format_number(controller['gate_pass_rate'])}",
        "- Gate pass rate after warmup: "
        f"{format_number(controller['non_warmup_gate_pass_rate'])}",
        f"- Final commit rate: {format_number(controller['commit_rate'])}",
        f"- Replacements: {controller['replacements']}",
        f"- Surprise-gate rejections: {controller['gate_rejections']}",
        f"- Priority rejections: {controller['priority_rejections']}",
        f"- Mean final bank size: {format_number(controller['final_bank_size_mean'])}",
        f"- Mean final threshold: {format_number(controller['final_threshold_mean'])}",
        f"- Mean routed frames per section: {format_number(controller['routed_frames_mean'])}",
        "- Mean routed frames actually used per section: "
        f"{format_number(controller['actually_retrieved_frames_mean'])}",
        "",
        "The controller rates describe writes to external memory. They are diagnostic "
        "statistics, not video-quality metrics.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize Surprise Forcing access traces."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_videos", type=int, default=None)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.jsonl"))
    if not paths:
        raise RuntimeError(f"No JSONL traces found in {args.input_dir}")
    rows = [summarize_trace(path) for path in paths]
    summary = build_summary(rows, expected_videos=args.expected_videos)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "per_video.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(args.output_dir / "report.md", summary)

    print(f"Wrote: {args.output_dir / 'summary.json'}")
    print(f"Wrote: {args.output_dir / 'per_video.csv'}")
    print(f"Wrote: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
