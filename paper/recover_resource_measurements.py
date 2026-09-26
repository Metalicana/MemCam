"""Export existing resource telemetry only. No Torch, model loading or generation."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
            f"Invalid resource measurement: {value}")
    return float(value)


def verify(path, expected, sources):
    actual = sha(path)
    require(actual == expected, f"Artifact hash mismatch: {path}")
    sources[str(path)] = actual


def summarize_profile(records, accesses, case):
    sections = [r for r in records if r.get("event") == "section_profile"]
    starts = [r for r in records if r.get("event") == "rollout_start"]
    ends = [r for r in records if r.get("event") == "rollout_summary"]
    count = (case["frames"] - 1) // 76
    require(len(starts) == len(ends) == 1 and ends[0].get("completed") is True,
            "Missing start or completed rollout summary")
    require([r["section_idx"] for r in sections] == list(range(count))
            and sections[-1]["section_end_frame"] == case["frames"] - 1,
            "Partial or duplicate profile sections")
    for record in records:
        require(record.get("memory_policy") == "slam_covisibility" and record.get("memory_budget") == 32
                and record.get("memory_bank_device") == "cpu"
                and record.get("scene") == case["start"]["scene"]
                and record.get("angle") == case["angle"] and record.get("num_frames") == case["frames"],
                "Profile identity differs from its case")
    require(all(r["stored_memory_size"] == 32 for r in sections), "Unexpected archive size")
    reads = [r for r in accesses if r.get("event") == "context_access"]
    expected = {(section, section * 76 + slot + 1) for section in range(count) for slot in range(76)}
    require(len(reads) == len(expected)
            and {(r["section_idx"], r["target_frame"]) for r in reads} == expected,
            "Missing or duplicated target reads")
    require(all(r.get("fallback_reason") == "initial_section" for r in reads if r["section_idx"] == 0),
            "Initial section is not the expected no-retrieval fallback")
    actual_reads = [r for r in reads if r["section_idx"] > 0]
    require(all(r.get("selected") is True and r["candidate_count"] > 0 for r in actual_reads),
            "Unexpected retrieval fallback")
    phases = {}
    for name in ("context_selection", "memory_policy_update", "bank_store"):
        expected_sections = sections[1:] if name == "context_selection" else sections
        require(all(name in r["phase_latency_s"] for r in expected_sections), f"Missing phase: {name}")
        phases[name] = sum(finite(r["phase_latency_s"][name]) for r in expected_sections)
    allocated = max(finite(r["peak_cuda_allocated_gb"]) for r in records)
    reserved = max(finite(r["peak_cuda_reserved_gb"]) for r in records)
    require(allocated <= reserved and allocated > 0, "Inconsistent CUDA peaks")
    rgb = max(int(r["bank_frame_bytes"]) for r in sections)
    require(rgb == int(ends[0]["peak_bank_frame_bytes"]), "Archive summary mismatch")
    return dict(case=case["id"], scene=case["start"]["scene"], angle=case["angle"],
                frames=case["frames"], generated_seconds=(case["frames"] - 1) / 30,
                archive_frames=32, archive_rgb_peak_mib=rgb / 2**20,
                archive_feature_logical_peak_mib=max(int(r["bank_feature_bytes"]) for r in sections) / 2**20,
                host_rss_sampled_peak_gib=max(finite(r["peak_rss_gb"]) for r in records),
                torch_allocated_rollout_peak_gib=allocated, torch_reserved_rollout_peak_gib=reserved,
                retrieval_queries=len(actual_reads), retrieval_total_s=phases["context_selection"],
                retrieval_ms_per_query=1000 * phases["context_selection"] / len(actual_reads),
                descriptor_and_update_total_s=phases["memory_policy_update"],
                descriptor_and_update_ms_per_section=1000 * phases["memory_policy_update"] / count,
                bank_store_total_ms=1000 * phases["bank_store"],
                rollout_s=finite(ends[0]["rollout_latency_s"]))


def roundtrip(root):
    plan = load(root / "plan.json")
    plan_sha = sha(root / "plan.json")
    case_signature = hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()
    sources = {str(root / "plan.json"): plan_sha}
    config = plan["generation"]
    require(all(config.get(k) == v for k, v in dict(policy="slam_covisibility", budget=32,
                steps=50, height=352, width=640, fps=30, bank_device="cpu", seed=42).items()),
            "Unexpected round-trip generation settings")
    require(len(plan["starts"]) == 5 and len({r["scene"] for r in plan["starts"]}) == 5
            and plan["tests"] == {"90": 153, "360": 609},
            "Unexpected round-trip cohort")
    device = load(root / "device_diagnostics.json")
    require(device.get("status") == "passed" and device.get("device_count") == 1,
            "No successful single-GPU diagnostics")
    sources[str(root / "device_diagnostics.json")] = sha(root / "device_diagnostics.json")
    rows = []
    for start in plan["starts"]:
        for angle, frames in ((90, 153), (360, 609)):
            case_id = f"{start['scene']}_{start['start_frame']:04d}_{angle}deg"
            receipt_path = root / "cases" / case_id / "generation.json"
            receipt = load(receipt_path)
            case = receipt["case"]
            require(receipt.get("status") == "complete" and case["id"] == case_id
                    and case["plan_sha256"] == case_signature and case["start"] == start
                    and case["angle"] == angle and case["frames"] == frames, "Mismatched generation receipt")
            attempt = receipt["attempt"]
            require(Path(attempt).name == attempt, "Invalid attempt path")
            directory = receipt_path.parent / attempt
            for name in ("profile.jsonl", "access.jsonl"):
                verify(directory / name, receipt["files"][name], sources)
            sources[str(receipt_path)] = sha(receipt_path)
            rows.append(summarize_profile(jsonl(directory / "profile.jsonl"), jsonl(directory / "access.jsonl"), case))
    summary = []
    for angle in (90, 360):
        subset = [r for r in rows if r["angle"] == angle]
        row = dict(angle=angle, videos=len(subset), frames=subset[0]["frames"],
                   generated_seconds=subset[0]["generated_seconds"], gpu=device["gpu_name"],
                   torch=device["torch"], cuda_build=device["torch_cuda_build"])
        for field in rows[0]:
            if field in ("case", "scene", "angle", "frames", "generated_seconds"):
                continue
            values = [r[field] for r in subset]
            row[field + ("_max" if "peak" in field else "_mean")] = max(values) if "peak" in field else mean(values)
        row["descriptor_and_update_ms_per_section_median"] = median(r["descriptor_and_update_ms_per_section"] for r in subset)
        summary.append(row)
    return rows, summary, dict(sources=sources, generation=config, device=device)


def proxy_updates(root):
    plan = load(root / "plan.json")
    require(plan.get("study") == "keepsake_component_proxy_cpu_v1", "Wrong proxy study")
    require(len(plan["items"]) == 15 and len({r["_row"] for r in plan["items"]}) == 15
            and len({r["scene"] for r in plan["items"]}) == 15, "Incomplete or duplicate proxy cohort")
    plan_sha = sha(root / "plan.json")
    sources, rows = {str(root / "plan.json"): plan_sha}, []
    for item in plan["items"]:
        directory = root / "cells" / f"row_{item['_row']:03d}"
        receipt = load(directory / "receipt.json")
        require(receipt.get("plan_sha256") == plan_sha and receipt.get("item") == item, "Wrong proxy receipt")
        hashes = receipt.get("artifacts", {})
        require(len(hashes) == 1 and Path(next(iter(hashes))).name == "replay.json", "Unexpected proxy artifacts")
        verify(directory / "replay.json", next(iter(hashes.values())), sources)
        sources[str(directory / "receipt.json")] = sha(directory / "receipt.json")
        updates = load(directory / "replay.json")["updates"]
        names = set(plan["configurations"])
        require(len(updates) == len(names) * 24
                and {(r["setting"], r["section_idx"]) for r in updates} == {(n, s) for n in names for s in range(24)},
                "Partial or duplicated CPU update timings")
        require(all(r["scene"] == item["scene"] and r["retained_count"] == 32 for r in updates), "Wrong proxy workload")
        for name in sorted(names):
            values = [finite(r["update_ms"]) for r in updates if r["setting"] == name]
            rows.append(dict(scene=item["scene"], setting=name, updates=24, mean_ms_per_update=mean(values),
                             total_s=sum(values) / 1000, host=receipt.get("host"), job=receipt.get("job")))
    summary = []
    for name in sorted(plan["configurations"]):
        subset = [r for r in rows if r["setting"] == name]
        summary.append(dict(setting=name, trajectories=len(subset), updates=sum(r["updates"] for r in subset),
                            mean_ms_per_update=mean(r["mean_ms_per_update"] for r in subset),
                            min_trajectory_mean_ms=min(r["mean_ms_per_update"] for r in subset),
                            max_trajectory_mean_ms=max(r["mean_ms_per_update"] for r in subset),
                            mean_total_s_per_history=mean(r["total_s"] for r in subset)))
    return rows, summary, dict(sources=sources, runtime=plan.get("runtime"),
        scope="Previously recorded single-pass CPU scorer + buffer update on cached histories; excludes extraction, "
              "RGB storage, reader and generation. Host/job retained per trajectory. Not matched end-to-end timing.")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latex(summary):
    environment = summary[0]
    hardware = str(environment["gpu"]).replace("_", r"\_")
    features = max(r["archive_feature_logical_peak_mib_max"] for r in summary)
    lines = [r"% Requires booktabs. Existing KEEPSAKE-only round-trip profiles, not a baseline comparison.",
             r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Recorded resources for KEEPSAKE-B32 with a CPU archive on " + hardware +
             f" (PyTorch {environment['torch']}, CUDA build {environment['cuda_build']}; " +
             r"640$\times$352, 50 denoising steps). "
             r"Five starting views per round-trip protocol. Memory columns are maxima across the five runs; "
             r"latencies are means of per-run averages. These are short-rollout diagnostics, not a matched "
             r"comparison to unbounded retention or measurements of the 60/180-second cohort.}",
             r"\label{tab:memcam-recorded-resources}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Protocol & RGB bank & Host RSS & GPU alloc. & GPU res. & Retrieval & Desc. + update \\",
             r" & (MiB) & (GiB)$^\dagger$ & (GiB) & (GiB) & (ms/query) & (ms/section) \\", r"\midrule"]
    for row in summary:
        lines.append(f"{row['angle']}$^\\circ$, {row['frames']} frames & " + " & ".join(f"{row[k]:.2f}" for k in (
            "archive_rgb_peak_mib_max", "host_rss_sampled_peak_gib_max", "torch_allocated_rollout_peak_gib_max",
            "torch_reserved_rollout_peak_gib_max", "retrieval_ms_per_query_mean", "descriptor_and_update_ms_per_section_mean")) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\begin{minipage}{\linewidth}\footnotesize",
              r"$^\dagger$Maximum observed RSS at profiler checkpoints, not a continuously measured host peak. "
              r"GPU columns use PyTorch rollout peak counters, not device-wide \texttt{nvidia-smi} usage. "
              r"Descriptor extraction and policy update share one timer; they cannot be separated retrospectively. "
              r"The RGB column excludes descriptors, metadata, output frames and model state. "
              f"The maximum recorded retained-feature logical payload is {features:.3f} MiB; this does not account for "
              r"Python containers or backing batches retained through array views. "
              r"Retrieval excludes the initial section's no-read fallback and includes no context encoding.",
              r"\end{minipage}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roundtrip-root", type=Path)
    parser.add_argument("--proxy-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.roundtrip_root or args.proxy_root, "Supply at least one existing results root")
    output = args.output.expanduser().resolve()
    roots = [p.expanduser().resolve() for p in (args.roundtrip_root, args.proxy_root) if p]
    require(all(output != r and not output.is_relative_to(r) and not r.is_relative_to(output) for r in roots),
            "Keep the report outside the input result roots")
    require(not output.exists(), "Choose a fresh report directory; existing reports are never overwritten")
    exports, provenance = {}, {}
    if args.roundtrip_root:
        rows, summary, provenance["roundtrip"] = roundtrip(args.roundtrip_root.expanduser().resolve())
        exports.update(roundtrip_per_video=rows, roundtrip_summary=summary)
    if args.proxy_root:
        rows, summary, provenance["proxy"] = proxy_updates(args.proxy_root.expanduser().resolve())
        exports.update(cpu_update_per_trajectory=rows, cpu_update_summary=summary)
    output.mkdir(parents=True)
    for name, rows in exports.items():
        write_csv(output / (name + ".csv"), rows)
    if "roundtrip_summary" in exports:
        (output / "resources.tex").write_text(latex(exports["roundtrip_summary"]))
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "status.json").write_text(json.dumps(dict(status="complete", mode="existing telemetry only",
                                                       new_videos=0, gpu_used=False), indent=2) + "\n")
    for name, rows in exports.items():
        if name.endswith("summary"):
            print(name + ": " + json.dumps(rows, indent=2))
    print(f"COMPLETE: {output}")


if __name__ == "__main__":
    main()
