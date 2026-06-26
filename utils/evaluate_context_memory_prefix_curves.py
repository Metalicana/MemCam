import argparse
import csv
import json
from pathlib import Path

from evaluate_context_memory import (
    BASE_METRIC_FIELDS,
    FVDRunner,
    LearnedMetricRunner,
    build_summary,
    evaluate_video,
    load_manifest,
    mean_or_none,
    output_path,
    parse_int_list,
    parse_learned_metrics,
    parse_rows,
    select_rows,
    write_csv,
    write_jsonl,
)


def make_prefix_item(item, prefix_duration):
    source_duration = int(item["duration_sec"])
    if prefix_duration > source_duration:
        raise ValueError(
            f"Cannot evaluate {prefix_duration}s prefix from {source_duration}s item "
            f"row {item.get('_row')}"
        )

    prefix_item = dict(item)
    source_frames = int(item["num_frames"])
    prefix_frames = int(round(source_frames * (prefix_duration / source_duration)))
    prefix_item["duration_sec"] = int(prefix_duration)
    prefix_item["num_frames"] = max(1, min(source_frames, prefix_frames))
    prefix_item["source_duration_sec"] = source_duration
    prefix_item["source_num_frames"] = source_frames
    return prefix_item


def status_only_result(item, model_output_dir, run_name):
    video_path = output_path(model_output_dir, item)
    status = "completed" if video_path.exists() else "missing_output"
    return {
        "run_name": run_name,
        "row": item["_row"],
        "status": status,
        "output": str(video_path),
        "scene": item["scene"],
        "start_frame": item["start_frame"],
        "duration_sec": item["duration_sec"],
        "source_duration_sec": item.get("source_duration_sec"),
        "num_frames_expected": item["num_frames"],
        "source_num_frames": item.get("source_num_frames"),
        "frames_seen": None,
        "frames_evaluated": 0,
        "frame_stride": None,
        "width": None,
        "height": None,
        "caption_key": item.get("caption_key"),
    }


def write_prefix_csv(path, rows, metric_fields):
    fieldnames = [
        "run_name",
        "row",
        "status",
        "scene",
        "start_frame",
        "duration_sec",
        "source_duration_sec",
        "num_frames_expected",
        "source_num_frames",
        "frames_seen",
        "frames_evaluated",
        "frame_stride",
        "width",
        "height",
        *metric_fields,
        "output",
        "caption_key",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_overall_from_durations(by_duration, metric_fields):
    overall = {
        "videos": sum(group.get("videos") or 0 for group in by_duration.values()),
        "completed_or_short": sum(
            group.get("completed_or_short") or 0 for group in by_duration.values()
        ),
        "missing_outputs": sum(group.get("missing_outputs") or 0 for group in by_duration.values()),
        "failed": sum(group.get("failed") or 0 for group in by_duration.values()),
        "frames_evaluated": sum(group.get("frames_evaluated") or 0 for group in by_duration.values()),
    }
    for field in metric_fields:
        overall[field] = mean_or_none([group.get(field) for group in by_duration.values()])
    if any(group.get("fvd") is not None for group in by_duration.values()):
        overall["fvd"] = mean_or_none([group.get("fvd") for group in by_duration.values()])
        overall["fvd_note"] = "Mean over prefix-duration FVD values; use by_duration for plots."
    return overall


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate duration-prefix curves from longer Context-as-Memory outputs. "
            "Example: compute 10/20/30/60s FVD/LPIPS from generated 60s videos."
        )
    )
    parser.add_argument("--manifest", type=Path, default=Path("testbeds/context_memory/manifest.jsonl"))
    parser.add_argument("--model_output_dir", type=Path, required=True)
    parser.add_argument("--metrics_dir", type=Path, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--source_duration", type=int, default=60)
    parser.add_argument("--eval_durations", type=str, default="10,20,30,60")
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--start_row", type=int, default=None)
    parser.add_argument("--end_row", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--frame_stride", type=int, default=30)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument(
        "--learned_metrics",
        type=str,
        default="lpips,fvd",
        help="Comma list: fvd, lpips, dino, clip. Use lpips,fvd for LPIPS plus FVD.",
    )
    parser.add_argument("--metric_device", type=str, default="cuda")
    parser.add_argument("--metric_batch_size", type=int, default=8)
    parser.add_argument("--learned_image_size", type=int, default=224)
    parser.add_argument("--fvd_clip_length", type=int, default=16)
    parser.add_argument("--fvd_clips_per_video", type=int, default=4)
    parser.add_argument("--fvd_frame_stride", type=int, default=4)
    parser.add_argument("--fvd_image_size", type=int, default=224)
    parser.add_argument("--fvd_backend", type=str, default="styleganv_i3d")
    parser.add_argument("--fvd_detector_path", type=Path, default=None)
    parser.add_argument("--fvd_detector_url", type=str, default=None)
    parser.add_argument("--fvd_cache_dir", type=Path, default=None)
    parser.add_argument("--no_fvd_download", action="store_false", dest="fvd_allow_download")
    parser.set_defaults(fvd_allow_download=True)
    parser.add_argument("--fvd_pca_dim", type=int, default=None)
    parser.add_argument("--fvd_eps", type=float, default=1e-6)
    parser.add_argument(
        "--write_frame_metrics",
        action="store_true",
        help="Write per-frame metrics when frame learned metrics such as LPIPS are requested.",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    eval_durations = parse_int_list(args.eval_durations)
    learned_metric_names = parse_learned_metrics(args.learned_metrics)
    fvd_requested = "fvd" in learned_metric_names
    frame_learned_metric_names = [
        metric_name for metric_name in learned_metric_names if metric_name != "fvd"
    ]
    compute_frame_metrics = bool(frame_learned_metric_names)

    learned_runner = None
    if frame_learned_metric_names:
        learned_runner = LearnedMetricRunner(
            metric_names=frame_learned_metric_names,
            device=args.metric_device,
            batch_size=args.metric_batch_size,
            image_size=args.learned_image_size,
        )

    fvd_kwargs = {
        "device": args.metric_device,
        "batch_size": args.metric_batch_size,
        "image_size": args.fvd_image_size,
        "clip_length": args.fvd_clip_length,
        "clips_per_video": args.fvd_clips_per_video,
        "frame_stride": args.fvd_frame_stride,
        "backend": args.fvd_backend,
        "detector_path": args.fvd_detector_path,
        "cache_dir": args.fvd_cache_dir,
        "allow_download": args.fvd_allow_download,
        "pca_dim": args.fvd_pca_dim,
        "eps": args.fvd_eps,
    }
    if args.fvd_detector_url:
        fvd_kwargs["detector_url"] = args.fvd_detector_url
    fvd_runner = FVDRunner(**fvd_kwargs) if fvd_requested else None

    metric_fields = BASE_METRIC_FIELDS + (
        learned_runner.fields if learned_runner is not None else []
    )
    if not compute_frame_metrics:
        metric_fields = []

    run_name = args.run_name or args.model_output_dir.name
    run_metrics_dir = args.metrics_dir / run_name
    run_metrics_dir.mkdir(parents=True, exist_ok=True)

    items = load_manifest(args.manifest)
    selected = select_rows(
        items=items,
        row_filter=parse_rows(args.rows),
        start_row=args.start_row,
        end_row=args.end_row,
        durations=[args.source_duration],
        limit=args.limit,
    )
    if not selected:
        raise RuntimeError(f"No manifest rows selected for source_duration={args.source_duration}.")

    metrics_jsonl = run_metrics_dir / "metrics.jsonl"
    metrics_csv = run_metrics_dir / "metrics.csv"
    summary_json = run_metrics_dir / "summary.json"
    frame_metrics_path = run_metrics_dir / "frame_metrics.jsonl"

    all_results = []
    prefix_items_by_duration = {}
    frame_metrics_handle = None
    if args.write_frame_metrics and compute_frame_metrics:
        frame_metrics_handle = frame_metrics_path.open("w", encoding="utf-8")

    try:
        for duration in eval_durations:
            prefix_items = [make_prefix_item(item, duration) for item in selected]
            prefix_items_by_duration[str(duration)] = prefix_items
            print(
                f"[{run_name}] evaluating {duration}s prefix from "
                f"{args.source_duration}s videos ({len(prefix_items)} manifest rows)"
            )

            if compute_frame_metrics:
                for item in prefix_items:
                    try:
                        result = evaluate_video(
                            item=item,
                            model_output_dir=args.model_output_dir,
                            dataset_root=args.dataset_root,
                            frame_stride=args.frame_stride,
                            max_frames=args.max_frames,
                            frame_metrics_handle=frame_metrics_handle,
                            learned_runner=learned_runner,
                            metric_fields=metric_fields,
                        )
                    except Exception as exc:
                        if args.strict:
                            raise
                        result = {
                            "row": item["_row"],
                            "status": "failed",
                            "error": repr(exc),
                            "output": str(output_path(args.model_output_dir, item)),
                            "scene": item["scene"],
                            "start_frame": item["start_frame"],
                            "duration_sec": item["duration_sec"],
                            "source_duration_sec": item.get("source_duration_sec"),
                            "num_frames_expected": item["num_frames"],
                            "source_num_frames": item.get("source_num_frames"),
                            "frames_evaluated": 0,
                        }
                    result["run_name"] = run_name
                    result["source_duration_sec"] = item.get("source_duration_sec")
                    result["source_num_frames"] = item.get("source_num_frames")
                    all_results.append(result)
            else:
                all_results.extend(
                    status_only_result(item, args.model_output_dir, run_name)
                    for item in prefix_items
                )
    finally:
        if frame_metrics_handle is not None:
            frame_metrics_handle.close()

    summary = build_summary(all_results, metric_fields)

    if fvd_runner is not None:
        for duration_text, prefix_items in prefix_items_by_duration.items():
            completed_rows = {
                row["row"]
                for row in all_results
                if str(row["duration_sec"]) == duration_text
                and row["status"] in {"completed", "short_video"}
            }
            duration_items = [item for item in prefix_items if item["_row"] in completed_rows]
            print(f"[{run_name}] computing FVD for {duration_text}s prefix")
            fvd_value, fvd_clips = fvd_runner.compute_group(
                items=duration_items,
                model_output_dir=args.model_output_dir,
                dataset_root=args.dataset_root,
                max_frames=args.max_frames,
            )
            summary["by_duration"][duration_text]["fvd"] = fvd_value
            summary["by_duration"][duration_text]["fvd_clips"] = fvd_clips
            summary["by_duration"][duration_text]["fvd_backend"] = fvd_runner.backend
            summary["by_duration"][duration_text]["fvd_detector_path"] = (
                str(fvd_runner.resolved_detector_path)
                if fvd_runner.resolved_detector_path is not None
                else None
            )

    summary["overall"] = summarize_overall_from_durations(summary["by_duration"], metric_fields)
    summary["metric_config"] = {
        "prefix_duration_curve": True,
        "source_duration": args.source_duration,
        "eval_durations": eval_durations,
        "learned_metrics": learned_metric_names,
        "frame_learned_metrics": frame_learned_metric_names,
        "video_distribution_metrics": ["fvd"] if fvd_requested else [],
        "compute_frame_metrics": compute_frame_metrics,
        "metric_device": args.metric_device,
        "metric_batch_size": args.metric_batch_size,
        "learned_image_size": args.learned_image_size,
        "fvd_clip_length": args.fvd_clip_length,
        "fvd_clips_per_video": args.fvd_clips_per_video,
        "fvd_frame_stride": args.fvd_frame_stride,
        "fvd_image_size": args.fvd_image_size,
        "fvd_backend": args.fvd_backend,
        "fvd_detector_path": str(args.fvd_detector_path) if args.fvd_detector_path else None,
        "fvd_cache_dir": str(args.fvd_cache_dir) if args.fvd_cache_dir else None,
        "fvd_allow_download": args.fvd_allow_download,
        "fvd_pca_dim_ignored": args.fvd_pca_dim,
        "fvd_eps": args.fvd_eps,
        "frame_stride": args.frame_stride if compute_frame_metrics else None,
        "max_frames": args.max_frames,
    }

    write_jsonl(metrics_jsonl, all_results)
    if compute_frame_metrics:
        write_csv(metrics_csv, all_results, metric_fields)
    else:
        write_prefix_csv(metrics_csv, all_results, metric_fields)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Wrote metrics: {metrics_jsonl}")
    print(f"Wrote CSV: {metrics_csv}")
    print(f"Wrote summary: {summary_json}")
    if args.write_frame_metrics and compute_frame_metrics:
        print(f"Wrote frame metrics: {frame_metrics_path}")


if __name__ == "__main__":
    main()
