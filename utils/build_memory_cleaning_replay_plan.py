"""Choose matched replay sections for the generated-memory cleaning test."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


FIELDS = (
    "selected_view_mismatch",
    "selected_memory_corruption",
    "selected_effective_mismatch",
    "candidate_count",
)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_sections(rows, run_name, duration, min_section, max_section):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("run_name") != run_name:
            continue
        if int(row["duration_sec"]) != int(duration):
            continue
        section_idx = int(row["section_idx"])
        if section_idx < int(min_section):
            continue
        if max_section is not None and section_idx > int(max_section):
            continue
        key = (
            int(row["row"]),
            row["scene"],
            int(row["dataset_start_frame"]),
            int(row["duration_sec"]),
            section_idx,
        )
        grouped[key].append(row)

    sections = []
    for key, values in sorted(grouped.items()):
        row_idx, scene, start_frame, duration_sec, section_idx = key
        selected_frames = {
            int(value["selected_memory_frame"])
            for value in values
            if value.get("selected_memory_frame") not in (None, "")
        }
        item = {
            "row": row_idx,
            "scene": scene,
            "dataset_start_frame": start_frame,
            "duration_sec": duration_sec,
            "section_idx": section_idx,
            "queries": len(values),
            "unique_selected_memory_frames": len(selected_frames),
        }
        for field in FIELDS:
            item[f"mean_{field}"] = float(
                np.mean([float(value[field]) for value in values])
            )
        sections.append(item)
    return sections


def select_cases(sections, count, unique_rows=True):
    ranked = sorted(
        sections,
        key=lambda row: (
            row["mean_selected_memory_corruption"],
            row["mean_selected_effective_mismatch"],
            row["section_idx"],
        ),
        reverse=True,
    )
    selected = []
    used_rows = set()
    for row in ranked:
        if unique_rows and row["row"] in used_rows:
            continue
        selected.append(
            {
                "case_index": len(selected),
                "selection_group": "high_selected_corruption",
                **row,
            }
        )
        used_rows.add(row["row"])
        if len(selected) >= int(count):
            break
    return selected


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_csv", type=Path, required=True)
    parser.add_argument("--run_name", default="baseline")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--min_section", type=int, default=20)
    parser.add_argument("--max_section", type=int, default=35)
    parser.add_argument("--max_cases", type=int, default=4)
    parser.add_argument("--allow_repeat_rows", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.min_section < 1:
        raise ValueError("--min_section must be at least 1")
    if args.max_section is not None and args.max_section < args.min_section:
        raise ValueError("--max_section must be greater than or equal to --min_section")
    if args.max_cases < 1:
        raise ValueError("--max_cases must be positive")

    sections = aggregate_sections(
        read_csv(args.query_csv),
        run_name=args.run_name,
        duration=args.duration,
        min_section=args.min_section,
        max_section=args.max_section,
    )
    cases = select_cases(
        sections,
        count=args.max_cases,
        unique_rows=not args.allow_repeat_rows,
    )
    if len(cases) < args.max_cases:
        raise RuntimeError(
            f"Only {len(cases)} eligible cases were found; requested {args.max_cases}"
        )

    write_csv(args.output, cases)
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "run_name": args.run_name,
                "duration": args.duration,
                "min_section": args.min_section,
                "max_section": args.max_section,
                "eligible_sections": len(sections),
                "selected_cases": len(cases),
                "selection": "highest mean selected-memory corruption, unique rows",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Eligible sections: {len(sections)}")
    for case in cases:
        print(
            f"case {case['case_index']}: row={case['row']} "
            f"section={case['section_idx']} "
            f"corruption={case['mean_selected_memory_corruption']:.4f} "
            f"view={case['mean_selected_view_mismatch']:.4f} "
            f"effective={case['mean_selected_effective_mismatch']:.4f}"
        )
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
