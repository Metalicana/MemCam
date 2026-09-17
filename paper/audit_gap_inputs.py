"""Audit the matched 60s gap sweep: logs and shared DINO caches, CPU only.

Does not submit jobs, generate videos, extract features, or change source files.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.analyze_retrieval_quality_decomposition import load_manifest, reconstruct_candidate_banks


CONFIGS = [("baseline", "unbounded", None)] + [
    (pattern.format(b=b), policy, b)
    for pattern, policy in (
        ("fifo_b{b}", "fifo"), ("ri_b{b}_dino_rgb", "rarity_irreplaceability"),
        ("slam_b{b}_covisibility", "slam_covisibility"),
        ("kcenter_b{b}", "kcenter_coreset"), ("mce_b{b}_lambda1_pilot", "mce"))
    for b in (16, 32, 64, 128)]


def audit_trace(path, item, policy, budget):
    sections = (int(item["num_frames"]) - 1) // 76
    if sections < 2 or (int(item["num_frames"]) - 1) % 76:
        raise ValueError("Expected whole 76-frame sections")
    expected = {(s, 76*s + slot + 1) for s in range(1, sections) for slot in range(76)}
    selected, evictions = {}, []
    hasher = hashlib.sha256()
    identity = (item["scene"], int(item["start_frame"]), int(item["duration_sec"]))
    with path.open("rb") as handle:
        for line in handle:
            hasher.update(line)
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("scene") is not None:
                observed = (event["scene"], int(event["dataset_start_frame"]), int(event["duration_sec"]))
                if observed != identity:
                    raise ValueError("Trace identity does not match manifest")
            if event.get("event") == "memory_eviction":
                evictions.append(event)
            if event.get("event") != "context_access" or not event.get("selected"):
                continue
            key = (int(event["section_idx"]), int(event["target_frame"]))
            if key in selected:
                raise ValueError(f"Duplicate selected read {key}")
            if event.get("memory_policy") != policy or (budget and int(event.get("memory_budget") or -1) != budget):
                raise ValueError("Policy/budget differs from the expected configuration")
            if event.get("selection_source", "retriever") != "retriever" or event.get("fallback_reason"):
                raise ValueError("Override or fallback read")
            selected[key] = event
    if set(selected) != expected:
        raise ValueError(f"Query coverage: missing={len(expected - set(selected))}, extra={len(set(selected) - expected)}")
    banks = reconstruct_candidate_banks(evictions, sections - 1, num_frames=int(item["num_frames"]))
    for (section, _), event in selected.items():
        bank = banks[section]
        if len(bank) != int(event.get("candidate_count", -1)):
            raise ValueError(f"Bank/count mismatch in section {section}")
        if int(event["selected_memory_frame"]) not in bank:
            raise ValueError(f"Selected index absent from bank in section {section}")
        if not bank or min(bank) < 0 or max(bank) >= section * 76 - 3:
            raise ValueError("Bank is not a subset of eligible history")
        if budget and (len(bank) > budget or int(event.get("stored_memory_size", budget)) > budget):
            raise ValueError("Recorded bank exceeds budget")
        if policy == "unbounded" and bank != list(range(section * 76 - 3)):
            raise ValueError("Unbounded history incomplete")
    return {"reads": len(selected), "sampled_reads": sum((target-s*76-1) % 19 == 0 for s, target in selected),
            "sha256": hasher.hexdigest()}


def audit_cache(path, expected_frames):
    features = np.load(path, mmap_mode="r", allow_pickle=False)
    if features.ndim != 2 or features.shape[0] < expected_frames or features.shape[1] != 768:
        raise ValueError(f"Expected >=({expected_frames},768) DINOv2-Base features; found {features.shape}")
    for start in range(0, expected_frames, 256):
        part = np.asarray(features[start:min(start+256, expected_frames)], dtype=np.float32)
        if not np.isfinite(part).all() or not np.allclose(np.linalg.norm(part, axis=1), 1, atol=.002, rtol=0):
            raise ValueError("Nonfinite or non-normalized features")
    return {"frames": int(features.shape[0]), "dimension": 768,
            "extra_frames": int(features.shape[0]) - expected_frames}


def inventory(args):
    items = load_manifest(args.manifest, duration=60)
    names = {i["output_prefix"] for i in items}
    if len(items) != args.expected_videos or len(names) != len(items):
        raise ValueError(f"Expected exactly {args.expected_videos} distinct 60s manifest entries; found {len(items)}")
    trace_rows, cache_rows, video_rows = [], [], []
    print("RUN                              VIDEOS   VALID TRACES", flush=True)
    for run, policy, budget in CONFIGS:
        ok = videos = 0
        for item in items:
            video = args.root / "context_memory_60s" / run / f"{item['output_prefix']}custom.mp4"
            present = video.is_file() and video.stat().st_size > 0
            videos += present
            video_rows.append(dict(run=run, row=item["_row"], path=str(video), present=present))
            path = video.parent / "access_traces" / f"{item['output_prefix']}custom.jsonl"
            record = dict(run=run, row=item["_row"], path=str(path))
            try:
                record.update(audit_trace(path, item, policy, budget), status="valid", error="")
                ok += 1
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record.update(status="missing" if not path.exists() else "invalid", error=str(exc))
            trace_rows.append(record)
        print(f"{run:32} {videos:2}/{len(items):2}     {ok:2}/{len(items):2}", flush=True)
    print("\nSearching for compatible GT and baseline DINO caches...", flush=True)
    stems = {f"{i['output_prefix']}dino.npy": i for i in items}
    roots = {args.feature_cache_dir} if args.feature_cache_dir else set()
    if args.feature_cache_dir is None:
        for path in args.root.rglob("*dino.npy"):
            if path.name in stems and path.parent.name in ("gt", "baseline"):
                roots.add(path.parent.parent)
    cache_summaries = []
    for cache_root in sorted(roots):
        counts = Counter()
        for kind in ("gt", "baseline"):
            for stem, item in stems.items():
                path = cache_root / kind / stem
                record = dict(root=str(cache_root), kind=kind, row=item["_row"], path=str(path))
                try:
                    record.update(audit_cache(path, int(item["num_frames"])), status="valid", error="")
                    counts[kind] += 1
                except (OSError, ValueError, TypeError, EOFError) as exc:
                    record.update(status="missing" if not path.exists() else "invalid", error=str(exc))
                cache_rows.append(record)
        cache_summaries.append(dict(root=str(cache_root), gt=counts["gt"], baseline=counts["baseline"]))
    if not roots:
        print("No compatible-name gt/baseline cache files found.")
    for summary in cache_summaries[:10]:
        print(f"GT {summary['gt']}/{len(items)}, baseline {summary['baseline']}/{len(items)}: {summary['root']}")
    if len(cache_summaries) > 10:
        print(f"{len(cache_summaries)-10} additional cache roots listed in audit.json")
    traces_ready = all(r["status"] == "valid" for r in trace_rows)
    complete_roots = [s["root"] for s in cache_summaries if s["gt"] == s["baseline"] == len(items)]
    print(f"\nAll policy traces valid: {'YES' if traces_ready else 'NO'}")
    print(f"Complete shared cache roots: {len(complete_roots)}")
    print("Cache checks cover names, dimensions, length and normalization, NOT encoder/video provenance.")
    return dict(cohort=[{k: i[k] for k in ("_row", "scene", "start_frame", "duration_sec", "num_frames", "output_prefix")} for i in items],
                traces=trace_rows, videos=video_rows, caches=cache_rows, cache_roots=cache_summaries,
                complete_cache_roots=complete_roots, all_traces_valid=traces_ready,
                manifest=str(args.manifest), manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                limitations="No source-video decoding or video/encoder cache provenance verification. No features extracted. Separate cache roots are not merged automatically.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "memcam_results")
    parser.add_argument("--manifest", type=Path, default=ROOT / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--expected-videos", type=int, default=15)
    parser.add_argument("--feature-cache-dir", type=Path, help="Inspect one known shared cache root instead of searching")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("Results root does not exist")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    report = inventory(args)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    issues = [dict(kind="trace", path=r["path"], error=r["error"]) for r in report["traces"] if r["status"] != "valid"]
    issues += [dict(kind="cache", path=r["path"], error=r["error"]) for r in report["caches"] if r["status"] != "valid"]
    with (args.output / "issues.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "path", "error"])
        writer.writeheader()
        writer.writerows(issues)
    print(f"Detailed audit: {args.output}")


if __name__ == "__main__":
    main()
