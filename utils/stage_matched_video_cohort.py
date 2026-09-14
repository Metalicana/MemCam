"""Stage an exact manifest-defined video cohort using symbolic links."""

import argparse
import json
from pathlib import Path


DEFAULT_RUNS = ["baseline"] + [
    pattern.format(budget)
    for pattern in (
        "fifo_b{}",
        "ri_b{}_dino_rgb",
        "slam_b{}_covisibility",
        "kcenter_b{}",
        "mce_b{}_lambda1_pilot",
    )
    for budget in (16, 32, 64, 128)
]


def load_cohort(manifest: Path, duration: int):
    cohort = []
    for row, line in enumerate(manifest.read_text().splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        if int(item["duration_sec"]) == duration:
            cohort.append((row, item))
    return cohort


def stage_run(source_root: Path, output_root: Path, run: str, cohort):
    destination = output_root / run
    destination.mkdir(parents=True, exist_ok=True)
    expected = set()
    for row, item in cohort:
        name = item["output_prefix"] + "custom.mp4"
        source = (source_root / run / name).resolve()
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"{run}, manifest row {row}: missing/empty {source}")
        target = destination / name
        expected.add(name)
        if target.is_symlink() and target.resolve() == source:
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to replace mismatched staged path: {target}")
        target.symlink_to(source)

    extras = sorted(
        path.name for path in destination.glob("*.mp4") if path.name not in expected
    )
    if extras:
        raise ValueError(f"{run}: staged directory contains extras: {extras}")
    present = [path for path in destination.glob("*.mp4") if path.is_file()]
    if len(present) != len(cohort):
        raise ValueError(f"{run}: expected {len(cohort)} staged videos, found {len(present)}")
    return len(present)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--expected-videos", type=int, default=15)
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    args = parser.parse_args()

    cohort = load_cohort(args.manifest, args.duration)
    if len(cohort) != args.expected_videos:
        raise ValueError(
            f"Expected {args.expected_videos} manifest rows at {args.duration}s, got {len(cohort)}"
        )
    runs = [run.strip() for run in args.runs.split(",") if run.strip()]
    if not runs:
        raise ValueError("No runs requested")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for run in runs:
        count = stage_run(args.source_root, args.output_root, run, cohort)
        print(f"{run:38s} {count}/{len(cohort)} EXACT")
    print(f"Staged {len(runs) * len(cohort)} run/video cells under {args.output_root}")


if __name__ == "__main__":
    main()
