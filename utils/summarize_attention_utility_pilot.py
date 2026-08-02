import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


PRIMARY_EFFECT = "prediction_relative_l2"
PREDICTORS = [
    "attention_total",
    "attention_per_slot",
    "slot_count",
    "retrieval_overlap_mean",
    "memory_age_mean",
]


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def rankdata(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + end - 1) / 2.0 + 1.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def pearson(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (x_value - left_mean) * (y_value - right_mean)
        for x_value, y_value in zip(left, right)
    )
    left_energy = sum((value - left_mean) ** 2 for value in left)
    right_energy = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_energy * right_energy)
    if denominator == 0:
        return None
    return numerator / denominator


def spearman(left, right):
    return pearson(rankdata(left), rankdata(right))


def finite_pairs(rows, predictor, effect):
    pairs = []
    for row in rows:
        x_value = row.get(predictor)
        y_value = row.get(effect)
        if x_value is None or y_value is None:
            continue
        x_value = float(x_value)
        y_value = float(y_value)
        if math.isfinite(x_value) and math.isfinite(y_value):
            pairs.append((x_value, y_value))
    return pairs


def probe_key(row):
    return (
        row.get("source_file"),
        row.get("row"),
        row.get("section_idx"),
        row.get("progress_id"),
    )


def predictor_summary(rows, predictor, effect=PRIMARY_EFFECT):
    pairs = finite_pairs(rows, predictor, effect)
    global_spearman = (
        spearman(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
        )
        if len(pairs) >= 2
        else None
    )

    grouped = defaultdict(list)
    for row in rows:
        grouped[probe_key(row)].append(row)
    within_values = []
    for group_rows in grouped.values():
        group_pairs = finite_pairs(group_rows, predictor, effect)
        if len(group_pairs) < 2:
            continue
        value = spearman(
            [pair[0] for pair in group_pairs],
            [pair[1] for pair in group_pairs],
        )
        if value is not None:
            within_values.append(value)

    return {
        "pairs": len(pairs),
        "global_spearman": global_spearman,
        "mean_within_probe_spearman": mean(within_values),
        "within_probe_groups": len(within_values),
    }


def role_summary(rows):
    grouped = defaultdict(dict)
    by_role = defaultdict(list)
    for row in rows:
        role = row.get("intervention_role")
        if role:
            grouped[probe_key(row)][role] = row
            by_role[role].append(row)

    comparisons = []
    for roles in grouped.values():
        top = roles.get("top_attention")
        bottom = roles.get("bottom_attention")
        if top is None or bottom is None:
            continue
        comparisons.append(
            float(top[PRIMARY_EFFECT]) - float(bottom[PRIMARY_EFFECT])
        )

    role_means = {}
    for role, role_rows in sorted(by_role.items()):
        role_means[role] = {
            "count": len(role_rows),
            "prediction_relative_l2_mean": mean(
                float(row[PRIMARY_EFFECT]) for row in role_rows
            ),
            "prediction_delta_mse_mean": mean(
                float(row["prediction_delta_mse"]) for row in role_rows
            ),
            "attention_total_mean": mean(
                float(row["attention_total"]) for row in role_rows
            ),
            "slot_count_mean": mean(float(row["slot_count"]) for row in role_rows),
        }

    wins = sum(value > 0 for value in comparisons)
    ties = sum(value == 0 for value in comparisons)
    return {
        "top_bottom_pairs": len(comparisons),
        "top_effect_greater_count": wins,
        "top_effect_equal_count": ties,
        "top_effect_greater_rate": (
            wins / len(comparisons) if comparisons else None
        ),
        "top_minus_bottom_relative_l2_mean": mean(comparisons),
        "by_role": role_means,
    }


def read_audits(input_dir, include_partial=False):
    rows = []
    complete_files = []
    partial_files = []
    for path in sorted(input_dir.glob("*.jsonl")):
        file_rows = []
        complete = False
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("event") == "attention_audit_complete":
                    complete = True
                elif payload.get("event") == "memory_intervention":
                    payload["source_file"] = path.name
                    file_rows.append(payload)
        if complete:
            complete_files.append(path.name)
            rows.extend(file_rows)
        else:
            partial_files.append(path.name)
            if include_partial:
                rows.extend(file_rows)
    return rows, complete_files, partial_files


def build_summary(rows, complete_files, partial_files):
    scenes = sorted({row.get("scene") for row in rows if row.get("scene")})
    probe_groups = {probe_key(row) for row in rows}
    return {
        "scope": {
            "complete_videos": len(complete_files),
            "partial_videos": len(partial_files),
            "scenes": len(scenes),
            "scene_names": scenes,
            "interventions": len(rows),
            "probe_groups": len(probe_groups),
            "complete_files": complete_files,
            "partial_files": partial_files,
        },
        "primary_effect": PRIMARY_EFFECT,
        "predictor_correlations": {
            predictor: predictor_summary(rows, predictor)
            for predictor in PREDICTORS
        },
        "role_comparison": role_summary(rows),
        "interpretation_limit": (
            "This pilot measures immediate denoiser sensitivity among memories already "
            "selected by the geometric retriever. It does not establish future utility "
            "for unretrieved memories or validate an eviction policy."
        ),
    }


def format_number(value):
    return "n/a" if value is None else f"{value:.4f}"


def write_report(path, summary):
    scope = summary["scope"]
    roles = summary["role_comparison"]
    lines = [
        "# Memory Attention Intervention Pilot",
        "",
        f"- Complete videos: {scope['complete_videos']}",
        f"- Intervention rows: {scope['interventions']}",
        f"- Section-step probe groups: {scope['probe_groups']}",
        "",
        "## Does attention predict intervention effect?",
        "",
        "| Predictor | Global Spearman | Mean within-probe Spearman | Groups |",
        "|---|---:|---:|---:|",
    ]
    for predictor in PREDICTORS:
        row = summary["predictor_correlations"][predictor]
        lines.append(
            f"| {predictor} | {format_number(row['global_spearman'])} | "
            f"{format_number(row['mean_within_probe_spearman'])} | "
            f"{row['within_probe_groups']} |"
        )
    lines.extend(
        [
            "",
            "## High-attention versus low-attention ablations",
            "",
            f"- Paired probes: {roles['top_bottom_pairs']}",
            "- High-attention memory caused the larger effect: "
            f"{format_number(roles['top_effect_greater_rate'])}",
            "- Mean high-minus-low relative effect: "
            f"{format_number(roles['top_minus_bottom_relative_l2_mean'])}",
            "",
            "## Reading this result",
            "",
            summary["interpretation_limit"],
            "",
            "The attention hypothesis is worth a larger future-loss experiment only if "
            "attention is consistently associated with intervention effect and is more "
            "informative than retrieval slot count and geometric overlap.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path, rows):
    fieldnames = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if not isinstance(value, (list, dict))
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(
        description="Summarize MemCam attention-versus-intervention pilot traces."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_videos", type=int, default=None)
    parser.add_argument("--include_partial", action="store_true")
    args = parser.parse_args()

    rows, complete_files, partial_files = read_audits(
        args.input_dir,
        include_partial=args.include_partial,
    )
    if args.expected_videos is not None and len(complete_files) != args.expected_videos:
        raise RuntimeError(
            f"Expected {args.expected_videos} complete audits, found {len(complete_files)}"
        )
    if not rows:
        raise RuntimeError("No memory_intervention rows found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(rows, complete_files, partial_files)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "interventions.csv", rows)
    write_report(args.output_dir / "report.md", summary)

    print(f"Complete audits: {len(complete_files)}")
    print(f"Interventions: {len(rows)}")
    attention = summary["predictor_correlations"]["attention_total"]
    controls = summary["predictor_correlations"]["slot_count"]
    print(
        "Attention Spearman: "
        f"global={format_number(attention['global_spearman'])}, "
        f"within={format_number(attention['mean_within_probe_spearman'])}"
    )
    print(
        "Slot-count control Spearman: "
        f"global={format_number(controls['global_spearman'])}, "
        f"within={format_number(controls['mean_within_probe_spearman'])}"
    )
    print(
        "Top-attention effect > bottom-attention effect: "
        f"{format_number(summary['role_comparison']['top_effect_greater_rate'])}"
    )
    print(f"Wrote: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
