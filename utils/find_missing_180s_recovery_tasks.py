import argparse
import json
import sys
from pathlib import Path


RECOVERY_RUNS = [
    "baseline",
    "fifo_b64",
    "ri_b64_dino_rgb",
    "ri_b128_dino_rgb",
]


def load_manifest(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Print Slurm array IDs for missing 180s recovery outputs."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    if len(rows) != 15:
        raise RuntimeError(f"Expected 15 manifest rows, found {len(rows)}")

    missing_ids = []
    for run_index, run_name in enumerate(RECOVERY_RUNS):
        run_dir = args.root / run_name
        missing_rows = []
        for row_index, item in enumerate(rows):
            output_path = run_dir / f"{item['output_prefix']}custom.mp4"
            if not output_path.exists():
                missing_ids.append(run_index * len(rows) + row_index)
                missing_rows.append(row_index)
        print(
            f"{run_name}: missing rows "
            f"{','.join(map(str, missing_rows)) if missing_rows else 'none'}",
            file=sys.stderr,
        )

    print(",".join(map(str, missing_ids)))


if __name__ == "__main__":
    main()
