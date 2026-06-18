import argparse
import csv
import json
import math
from pathlib import Path


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def rank_values(values, reverse=True):
    order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=reverse)
    ranks = [0] * len(values)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 1e-12 or y_var <= 1e-12:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return cov / math.sqrt(x_var * y_var)


def spearman(xs, ys):
    if len(xs) < 2:
        return None
    return pearson(rank_values(xs), rank_values(ys))


def topk_overlap(rows, score_key, gt_key, k):
    if not rows:
        return None
    k = min(k, len(rows))
    top_score = {
        row["frame_idx"]
        for row in sorted(rows, key=lambda row: row[score_key], reverse=True)[:k]
    }
    top_gt = {
        row["frame_idx"]
        for row in sorted(rows, key=lambda row: row[gt_key], reverse=True)[:k]
    }
    return len(top_score & top_gt) / k if k else None


def gt_topk_kept(rows, gt_key, k):
    if not rows:
        return None
    k = min(k, len(rows))
    top_gt = sorted(rows, key=lambda row: row[gt_key], reverse=True)[:k]
    return sum(bool(row["kept_after"]) for row in top_gt) / k if k else None


def gt_mass_kept(rows, gt_key):
    total = sum(float(row[gt_key] or 0) for row in rows)
    if total <= 0:
        return None
    kept = sum(float(row[gt_key] or 0) for row in rows if row["kept_after"])
    return kept / total


def group_rows(rows):
    groups = {}
    for row in rows:
        key = (row["row"], row["scene"], row["section_idx"], row["budget"])
        groups.setdefault(key, []).append(row)
    return groups


def summarize_decision(key, rows, topk):
    run_row, scene, section_idx, budget = key
    ri_scores = [float(row["ri_score"]) for row in rows]
    gt_future = [float(row["gt_future_use_count"] or 0) for row in rows]
    gt_horizon = [float(row["gt_horizon_use_count"] or 0) for row in rows]

    return {
        "row": run_row,
        "scene": scene,
        "section_idx": section_idx,
        "budget": budget,
        "candidates": len(rows),
        "kept": sum(bool(row["kept_after"]) for row in rows),
        "evicted": sum(bool(row["evicted"]) for row in rows),
        "spearman_ri_vs_gt_future": spearman(ri_scores, gt_future),
        "spearman_ri_vs_gt_horizon": spearman(ri_scores, gt_horizon),
        "topk_overlap_future": topk_overlap(rows, "ri_score", "gt_future_use_count", topk),
        "topk_overlap_horizon": topk_overlap(rows, "ri_score", "gt_horizon_use_count", topk),
        "gt_future_topk_kept": gt_topk_kept(rows, "gt_future_use_count", topk),
        "gt_horizon_topk_kept": gt_topk_kept(rows, "gt_horizon_use_count", topk),
        "gt_future_mass_kept": gt_mass_kept(rows, "gt_future_use_count"),
        "gt_horizon_mass_kept": gt_mass_kept(rows, "gt_horizon_use_count"),
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    metric_keys = [
        "spearman_ri_vs_gt_future",
        "spearman_ri_vs_gt_horizon",
        "topk_overlap_future",
        "topk_overlap_horizon",
        "gt_future_topk_kept",
        "gt_horizon_topk_kept",
        "gt_future_mass_kept",
        "gt_horizon_mass_kept",
    ]
    return {
        "decisions": len(rows),
        **{key: mean([row[key] for row in rows]) for key in metric_keys},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize RI score alignment against overlap-label future usefulness."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("/data/ab575577/MemCam/analysis/context_memory/ri_frame_scores.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/data/ab575577/MemCam/analysis/context_memory"),
    )
    parser.add_argument("--topk", type=int, default=32)
    args = parser.parse_args()

    rows = load_jsonl(args.scores)
    if not rows:
        raise RuntimeError(f"No rows found in {args.scores}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    decision_rows = [
        summarize_decision(key, group, topk=args.topk)
        for key, group in sorted(group_rows(rows).items())
    ]
    summary = aggregate(decision_rows)

    write_csv(args.output_dir / "ri_alignment_by_decision.csv", decision_rows)
    with (args.output_dir / "ri_alignment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {args.output_dir / 'ri_alignment_by_decision.csv'}")
    print(f"Wrote: {args.output_dir / 'ri_alignment_summary.json'}")


if __name__ == "__main__":
    main()
