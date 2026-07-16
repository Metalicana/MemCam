import argparse
import csv
import json
from pathlib import Path


DEFAULT_RUNS = (
    "baseline,"
    "fifo_b16,fifo_b32,fifo_b64,fifo_b128,"
    "ri_b16_dino_rgb,ri_b32_dino_rgb,ri_b64_dino_rgb,ri_b128_dino_rgb,"
    "slam_b16_covisibility,slam_b32_covisibility,"
    "slam_b64_covisibility,slam_b128_covisibility,"
    "kcenter_b16_dino_pose,kcenter_b32_dino_pose,"
    "kcenter_b64_dino_pose,kcenter_b128_dino_pose"
)


def parse_runs(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_by_key(path, key, secondary_filter=None):
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if secondary_filter is not None:
        rows = [row for row in rows if secondary_filter(row)]
    return {row[key]: row for row in rows}


def optional_float(row, key):
    if not row:
        return None
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def optional_int(row, key):
    if not row:
        return None
    value = row.get(key)
    if value in (None, ""):
        return None
    return int(float(value))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def main():
    parser = argparse.ArgumentParser(
        description="Join final 180s quality, CUT3R, and revisit summaries."
    )
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--cut3r_summary", type=Path, required=True)
    parser.add_argument("--revisit_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--runs", type=str, default=DEFAULT_RUNS)
    parser.add_argument("--expected_videos", type=int, default=15)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    runs = parse_runs(args.runs)
    cut3r = load_csv_by_key(args.cut3r_summary, "run_name")
    revisit_delta = load_csv_by_key(
        args.revisit_dir / "tables" / "revisit_delta_summary_vs_gt.csv",
        "source",
        secondary_filter=lambda row: row.get("revisit_type") == "exact_pose",
    )
    revisit_self = load_csv_by_key(
        args.revisit_dir / "tables" / "revisit_summary.csv",
        "source",
        secondary_filter=lambda row: row.get("revisit_type") == "exact_pose",
    )

    output_rows = []
    errors = []
    for run in runs:
        summary_path = args.eval_dir / run / "summary.json"
        if not summary_path.exists():
            errors.append(f"missing quality summary: {summary_path}")
            quality = {}
        else:
            quality = load_json(summary_path).get("overall", {})

        quality_videos = optional_int(quality, "videos")
        quality_complete = optional_int(quality, "completed_or_short")
        if quality_videos != args.expected_videos or quality_complete != args.expected_videos:
            errors.append(
                f"{run}: quality coverage is {quality_complete}/{quality_videos}, "
                f"expected {args.expected_videos}/{args.expected_videos}"
            )

        cut3r_row = cut3r.get(run)
        cut3r_videos = optional_int(cut3r_row, "videos")
        if cut3r_videos != args.expected_videos:
            errors.append(
                f"{run}: CUT3R coverage is {cut3r_videos}, expected {args.expected_videos}"
            )

        delta_row = revisit_delta.get(run)
        self_row = revisit_self.get(run)
        output_rows.append(
            {
                "run_name": run,
                "quality_videos": quality_complete,
                "fvd": optional_float(quality, "fvd"),
                "lpips_alex": optional_float(quality, "lpips_alex"),
                "dino_distance": optional_float(quality, "dino_distance"),
                "psnr_db": optional_float(quality, "psnr_db"),
                "ssim": optional_float(quality, "ssim"),
                "cut3r_videos": cut3r_videos,
                "cut3r_rotation_error_deg": optional_float(
                    cut3r_row, "rotation_error_deg_mean_mean"
                ),
                "cut3r_translation_error": optional_float(
                    cut3r_row, "translation_error_scale_only_mean_mean"
                ),
                "worldscore_camera_control": optional_float(
                    cut3r_row, "worldscore_camera_control_score_mean"
                ),
                "revisit_videos": optional_int(delta_row, "videos"),
                "revisit_clusters": optional_int(delta_row, "clusters"),
                "revisit_mean_delta_rmse": optional_float(
                    delta_row, "mean_delta_rmse"
                ),
                "revisit_mean_worst_delta_rmse": optional_float(
                    delta_row, "mean_worst_delta_rmse"
                ),
                "revisit_self_mean_patch_rmse": optional_float(
                    self_row, "mean_patch_rmse"
                ),
            }
        )

    if errors and args.strict:
        raise RuntimeError("\n".join(errors))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "report_summary.csv"
    write_csv(csv_path, output_rows)

    report_path = args.output_dir / "report.md"
    columns = [
        ("run_name", "run"),
        ("fvd", "FVD"),
        ("lpips_alex", "LPIPS"),
        ("dino_distance", "DINO dist"),
        ("cut3r_rotation_error_deg", "CUT3R Rdeg"),
        ("cut3r_translation_error", "CUT3R T"),
        ("revisit_videos", "revisit videos"),
        ("revisit_mean_delta_rmse", "revisit delta RMSE"),
    ]
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# MemCam 180s Final Metrics\n\n")
        handle.write("| " + " | ".join(label for _, label in columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in output_rows:
            handle.write(
                "| "
                + " | ".join(
                    str(row[key]) if key == "run_name" else fmt(row[key])
                    for key, _ in columns
                )
                + " |\n"
            )
        handle.write("\n")
        handle.write(
            "Revisit values use exact-pose events, a 15-second minimum gap, and the "
            "GT-oracle worst-patch RMSE <= 50 filter. Fewer than 15 revisit videos is "
            "expected when a trajectory has no qualifying event.\n"
        )
        if errors:
            handle.write("\n## Coverage Warnings\n\n")
            for error in errors:
                handle.write(f"- {error}\n")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {report_path}")
    if errors:
        print("Coverage warnings:")
        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()
