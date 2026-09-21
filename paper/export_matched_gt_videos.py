"""Export the frozen matched GT cohort as lossless RGB FFV1 clips, CPU only."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def load_items(plan_path, expected_videos):
    plan = json.loads(plan_path.read_text())
    manifest = Path(plan["manifest"])
    if digest(manifest) != plan["manifest_sha256"]:
        raise ValueError("Frozen manifest hash does not match the metric plan")
    lines, rows = manifest.read_text().splitlines(), plan["rows"]
    if (len(rows) != expected_videos or len(set(rows)) != len(rows)
            or any(type(row) is not int or not 0 <= row < len(lines) for row in rows)):
        raise ValueError("Plan must identify the expected distinct manifest rows")
    items = [dict(json.loads(lines[row]), manifest_row=row) for row in rows]
    names = [item["output_prefix"] + "custom.mp4" for item in items]
    if len(set(names)) != len(names) or sorted(names) != sorted(plan["expected"]):
        raise ValueError("GT cohort does not match the plan's generated-video identities")
    for item in items:
        if (item["duration_sec"] != 60 or int(item["start_frame"]) < 0
                or int(item["num_frames"]) < 1 or Fraction(str(item["fps"])) <= 0
                or Path(item["output_prefix"]).name != item["output_prefix"]):
            raise ValueError("Invalid 60-second cohort item")
    return items


def source_signature(item):
    # File stats detect ordinary source changes without hashing every full PNG.
    # This is explicitly not a source-pixel content hash.
    folder = Path(item["gt_frames_dir"]).resolve()
    signature = hashlib.sha256(str(folder).encode())
    for index in range(int(item["start_frame"]), int(item["start_frame"]) + int(item["num_frames"])):
        path = folder / f"{index:04d}.png"
        stat = path.stat()
        if not path.is_file() or stat.st_size == 0:
            raise ValueError(f"Missing or empty GT image: {path}")
        signature.update(f"\n{index},{stat.st_size},{stat.st_mtime_ns}".encode())
    return signature.hexdigest()


def encode_command(ffmpeg, item, output, threads, count=None):
    return [ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-y",
            "-framerate", str(item["fps"]), "-start_number", str(item["start_frame"]),
            "-i", str(Path(item["gt_frames_dir"]) / "%04d.png"),
            "-frames:v", str(count or item["num_frames"]), "-an",
            "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0",
            "-threads", str(threads), str(output)]


def verify_video(ffprobe, path, item, count=None):
    result = subprocess.run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=codec_name,pix_fmt,nb_read_frames,avg_frame_rate,width,height",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream: {path}")
    stream = streams[0]
    if (stream["codec_name"] != "ffv1" or stream["pix_fmt"] != "bgr0"
            or int(stream["nb_read_frames"]) != (count or int(item["num_frames"]))
            or Fraction(stream["avg_frame_rate"]) != Fraction(str(item["fps"]))):
        raise ValueError(f"Wrong GT clip format, frame count or FPS: {path}: {stream}")
    return stream


def export(plan, output, expected_videos=15, threads=4, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    if threads < 1:
        raise ValueError("threads must be positive")
    ffmpeg, ffprobe = shutil.which(ffmpeg), shutil.which(ffprobe)
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError("Both ffmpeg and ffprobe must be available; load the ffmpeg module")
    print(f"FFmpeg: {ffmpeg}\nFFprobe: {ffprobe}", flush=True)
    help_result = subprocess.run([ffmpeg, "-hide_banner", "-h", "encoder=ffv1"],
                                 check=True, capture_output=True, text=True)
    help_text = help_result.stdout + help_result.stderr
    if "Encoder ffv1" not in help_text or "bgr0" not in help_text:
        raise RuntimeError(f"{ffmpeg} lacks the FFV1 RGB encoder; use --ffmpeg with another build")
    items = load_items(plan, expected_videos)
    sources = [source_signature(item) for item in items]
    output.mkdir(parents=True, exist_ok=True)
    # Exercise the selected executable on actual GT before starting the export.
    with tempfile.TemporaryDirectory(dir=output, prefix="encoder_check_") as temporary:
        check = Path(temporary) / "check.mkv"
        subprocess.run(encode_command(ffmpeg, items[0], check, threads, count=1), check=True)
        verify_video(ffprobe, check, items[0], count=1)
    version = subprocess.run([ffmpeg, "-version"], check=True, capture_output=True, text=True).stdout
    files, records = [], []
    for number, (item, signature) in enumerate(zip(items, sources), 1):
        video = output / (item["output_prefix"] + "gt.mkv")
        receipt = video.with_suffix(".json")
        inputs = dict(item=item, source_signature=signature, codec="ffv1", pix_fmt="bgr0", level=3)
        previous = json.loads(receipt.read_text()) if receipt.exists() else {}
        reusable = (previous.get("inputs") == inputs and video.is_file()
                    and previous.get("video_sha256") == digest(video))
        if reusable:
            print(f"[{number}/{len(items)}] KEEP {video.name}", flush=True)
        else:
            print(f"[{number}/{len(items)}] ENCODE {video.name}", flush=True)
            partial = video.with_name(video.stem + ".partial.mkv")
            subprocess.run(encode_command(ffmpeg, item, partial, threads), check=True)
            stream = verify_video(ffprobe, partial, item)
            if source_signature(item) != signature:
                raise ValueError(f"GT sources changed during encoding: {video.name}")
            partial.replace(video)
            previous = dict(inputs=inputs, video_sha256=digest(video), stream=stream,
                            ffmpeg=ffmpeg, ffmpeg_version=version.splitlines()[0],
                            source_signature_kind="directory and per-frame index, size, mtime_ns; not pixel hashes")
            write_json(receipt, previous)
        records.append(dict(item, gt_video=video.name, video_sha256=previous["video_sha256"],
                            frame_mapping="GT video frame n maps to dataset frame start_frame + n"))
        files.extend((video, receipt))
    manifest = output / "manifest.json"
    write_json(manifest, records)
    files.append(manifest)
    archive = output.with_suffix(".zip")
    partial_archive = archive.with_suffix(".partial.zip")
    with zipfile.ZipFile(partial_archive, "w", compression=zipfile.ZIP_STORED) as handle:
        for path in files:
            handle.write(path, path.name)
    partial_archive.replace(archive)
    print(f"Saved {len(items)} verified GT videos: {archive}\nSize: {archive.stat().st_size / 2**30:.2f} GiB", flush=True)
    return archive


def main():
    root = Path.home() / "memcam_results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=root / "budget_metrics_60s_820776/plan.json")
    parser.add_argument("--output", type=Path, default=root / "gt_matched15_60s_ffv1")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--expected-videos", type=int, default=15)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    export(args.plan, args.output, args.expected_videos, args.threads, args.ffmpeg, args.ffprobe)


if __name__ == "__main__":
    main()
