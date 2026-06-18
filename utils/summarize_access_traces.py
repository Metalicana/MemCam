import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


AGE_BINS = [
    ("0_75", 0, 75),
    ("76_151", 76, 151),
    ("152_303", 152, 303),
    ("304_plus", 304, None),
]


def iter_trace_rows(trace_dir):
    for path in sorted(trace_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_trace_file"] = path.name
                yield row


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values, q):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * q
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return values[low] * (1 - weight) + values[high] * weight


def gini(values):
    values = sorted(value for value in values if value is not None)
    if not values or sum(values) == 0:
        return None
    n = len(values)
    weighted_sum = sum((idx + 1) * value for idx, value in enumerate(values))
    return (2 * weighted_sum) / (n * sum(values)) - (n + 1) / n


def age_bin(age):
    for name, low, high in AGE_BINS:
        if age >= low and (high is None or age <= high):
            return name
    return "unknown"


def safe_round(value, digits=6):
    if value is None:
        return None
    return round(value, digits)


def group_key(row):
    return (
        row.get("run_memory_policy") or row.get("memory_policy") or "unknown",
        row.get("run_memory_budget", row.get("memory_budget")),
        row.get("duration_sec"),
    )


def summarize_group(key, rows):
    policy, budget, duration_sec = key
    access_rows = [row for row in rows if row.get("event") == "context_access"]
    selected_rows = [row for row in access_rows if row.get("selected")]
    fallback_rows = [row for row in access_rows if not row.get("selected")]
    ages = [int(row["memory_age"]) for row in selected_rows if row.get("memory_age") is not None]
    overlaps = [
        float(row["selected_overlap"])
        for row in selected_rows
        if row.get("selected_overlap") is not None
    ]
    selected_frame_counts = Counter(
        (
            row.get("row"),
            row.get("scene"),
            row.get("selected_memory_frame"),
        )
        for row in selected_rows
    )
    reuse_counts = list(selected_frame_counts.values())
    bin_counts = Counter(age_bin(age) for age in ages)

    summary = {
        "memory_policy": policy,
        "memory_budget": budget,
        "duration_sec": duration_sec,
        "trace_files": len({row["_trace_file"] for row in rows}),
        "queries": len(access_rows),
        "selected_queries": len(selected_rows),
        "fallback_queries": len(fallback_rows),
        "fallback_rate": len(fallback_rows) / len(access_rows) if access_rows else 0.0,
        "unique_selected_frames": len(selected_frame_counts),
        "mean_age": safe_round(mean(ages)),
        "median_age": safe_round(percentile(ages, 0.5)),
        "p90_age": safe_round(percentile(ages, 0.9)),
        "p95_age": safe_round(percentile(ages, 0.95)),
        "mean_overlap": safe_round(mean(overlaps)),
        "median_overlap": safe_round(percentile(overlaps, 0.5)),
        "max_reuse_count": max(reuse_counts) if reuse_counts else 0,
        "mean_reuse_count": safe_round(mean(reuse_counts)),
        "reuse_gini": safe_round(gini(reuse_counts)),
    }
    for name, _, _ in AGE_BINS:
        count = bin_counts[name]
        summary[f"age_bin_{name}"] = count
        summary[f"age_bin_{name}_frac"] = count / len(ages) if ages else 0.0
    return summary


def selected_frame_rows(rows):
    counts = Counter()
    overlap_sums = defaultdict(float)
    age_sums = defaultdict(float)
    metadata = {}
    for row in rows:
        if row.get("event") != "context_access" or not row.get("selected"):
            continue
        key = (
            row.get("run_memory_policy") or row.get("memory_policy") or "unknown",
            row.get("run_memory_budget", row.get("memory_budget")),
            row.get("row"),
            row.get("scene"),
            row.get("selected_memory_frame"),
        )
        counts[key] += 1
        overlap_sums[key] += float(row.get("selected_overlap") or 0.0)
        age_sums[key] += float(row.get("memory_age") or 0.0)
        metadata[key] = {
            "duration_sec": row.get("duration_sec"),
            "dataset_start_frame": row.get("dataset_start_frame"),
        }

    output = []
    for key, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        policy, budget, row_id, scene, frame_idx = key
        output.append(
            {
                "memory_policy": policy,
                "memory_budget": budget,
                "row": row_id,
                "scene": scene,
                "selected_memory_frame": frame_idx,
                "selected_dataset_frame": (
                    int(metadata[key]["dataset_start_frame"]) + int(frame_idx)
                    if metadata[key].get("dataset_start_frame") is not None and frame_idx is not None
                    else None
                ),
                "duration_sec": metadata[key]["duration_sec"],
                "access_count": count,
                "mean_overlap": safe_round(overlap_sums[key] / count),
                "mean_age": safe_round(age_sums[key] / count),
            }
        )
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize MemCam memory access traces emitted during generation."
    )
    parser.add_argument("--trace_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    rows = list(iter_trace_rows(args.trace_dir))
    if not rows:
        raise RuntimeError(f"No trace rows found in {args.trace_dir}")

    output_dir = args.output_dir or args.trace_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    summary_rows = [
        summarize_group(key, group_rows)
        for key, group_rows in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]
    frame_rows = selected_frame_rows(rows)

    write_csv(output_dir / "access_summary.csv", summary_rows)
    write_csv(output_dir / "access_selected_frames.csv", frame_rows)
    with (output_dir / "access_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)
        handle.write("\n")

    print(json.dumps(summary_rows, indent=2))
    print(f"Wrote: {output_dir / 'access_summary.csv'}")
    print(f"Wrote: {output_dir / 'access_summary.json'}")
    print(f"Wrote: {output_dir / 'access_selected_frames.csv'}")


if __name__ == "__main__":
    main()
