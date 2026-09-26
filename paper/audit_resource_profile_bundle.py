"""Audit saved resource logs without extraction, model loading, or new runs."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from statistics import mean
import tarfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
            f"Invalid measurement: {value!r}")
    return float(value)


def summarize_profile(records, member):
    starts = [r for r in records if r.get("event") == "rollout_start"]
    sections = [r for r in records if r.get("event") == "section_profile"]
    ends = [r for r in records if r.get("event") == "rollout_summary"]
    require(len(starts) == 1, f"Expected one rollout: {member}")
    start = starts[0]
    identity = ("scene", "dataset_start_frame", "num_frames", "total_sections",
                "memory_policy", "memory_budget", "memory_bank_device")
    require(all(all(r.get(k) == start.get(k) for k in identity) for r in records),
            f"Inconsistent profile identity: {member}")
    require([r["section_idx"] for r in sections] == list(range(len(sections))),
            f"Missing or duplicated sections: {member}")
    row = dict(member=member, run=member.split("/profiles/")[0],
               scene=start.get("scene"), start_frame=start.get("dataset_start_frame"),
               policy=start["memory_policy"], budget=start.get("memory_budget"),
               bank_device=start["memory_bank_device"], duration_sec=start["duration_sec"],
               frames=start["num_frames"], observed_sections=len(sections),
               expected_sections=start["total_sections"], status="incomplete")
    if not ends:
        return row
    require(len(ends) == 1, f"Multiple rollout summaries: {member}")
    if ends[0].get("completed") is not True:
        return row
    require(len(sections) == start["total_sections"] and sections
            and sections[-1]["section_end_frame"] == start["num_frames"] - 1
            and ends[0]["sections"] == len(sections), f"False completion: {member}")
    rgb = max(number(r["bank_frame_bytes"]) for r in sections)
    require(rgb == number(ends[0]["peak_bank_frame_bytes"]), f"Archive mismatch: {member}")
    row.update(status="complete", archive_frames_max=max(r["stored_memory_size"] for r in sections),
               archive_rgb_peak_mib=rgb / 2**20,
               archive_features_logical_peak_mib=max(number(r["bank_feature_bytes"]) for r in sections) / 2**20,
               host_rss_sampled_peak_gib=max(number(r["peak_rss_gb"]) for r in records),
               torch_allocated_rollout_peak_gib=max(number(r["peak_cuda_allocated_gb"]) for r in records),
               torch_reserved_rollout_peak_gib=max(number(r["peak_cuda_reserved_gb"]) for r in records),
               rollout_s=number(ends[0]["rollout_latency_s"]))
    require(row["torch_allocated_rollout_peak_gib"] <= row["torch_reserved_rollout_peak_gib"],
            f"Inconsistent CUDA peaks: {member}")
    for phase in ("context_selection", "memory_policy_update", "bank_store"):
        # The initial section has no retrieval, so do not count it as a timed reader call.
        timed = sections[1:] if phase == "context_selection" else sections
        require(timed and all(phase in r["phase_latency_s"] for r in timed),
                f"Missing {phase}: {member}")
        total = sum(number(r["phase_latency_s"][phase]) for r in timed)
        row[phase + "_total_s"] = total
        row[phase + "_mean_s_per_section"] = total / len(timed)
    return row


def match_elapsed(baseline, keepsake, duration, cohort="seed0_"):
    arms = []
    for records, policy, budget in ((baseline, "unbounded", None),
                                     (keepsake, "slam_covisibility", 32)):
        found = {}
        for r in records:
            name = PurePosixPath(r.get("output", "")).name
            if r.get("status") != "completed" or not name.startswith(cohort):
                continue
            require(name not in found, f"Ambiguous repeated completion: {name}")
            require(r.get("memory_policy") == policy and r.get("memory_budget") == budget
                    and r.get("duration_sec") == duration and r.get("steps") == 50,
                    f"Unexpected elapsed-time protocol: {name}")
            number(r["time_sec"])
            found[name] = r
        arms.append(found)
    require(set(arms[0]) == set(arms[1]) and len(arms[0]) == 15,
            f"Expected fifteen matched identities for {duration}s/{cohort}")
    result = []
    for name in sorted(arms[0]):
        a, b = (arm[name] for arm in arms)
        require(a["num_frames"] == b["num_frames"], f"Frame mismatch: {name}")
        result.append(dict(duration_sec=duration, cohort_prefix=cohort, output_basename=name,
                           frames=a["num_frames"], steps=50, unbounded_elapsed_s=a["time_sec"],
                           keepsake_b32_elapsed_s=b["time_sec"], hardware_matching="unverified"))
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(bundle, output):
    require(not output.exists(), f"Refusing to overwrite {output}")
    profiles, statuses, hashes = [], {}, {}
    with tarfile.open(bundle, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = member.name.removeprefix("./")
            is_profile = "/profiles/" in name and name.endswith(".jsonl")
            is_status = name.endswith("/run_status.jsonl")
            if not (is_profile or is_status):
                continue
            require(name not in hashes, f"Duplicate archive member: {name}")
            data = archive.extractfile(member).read()
            hashes[name] = hashlib.sha256(data).hexdigest()
            records = [json.loads(line) for line in data.splitlines() if line.strip()]
            if is_profile:
                profiles.append(summarize_profile(records, name))
            else:
                statuses[name] = records
    groups = defaultdict(list)
    for row in profiles:
        groups[row["run"]].append(row)
    summaries = []
    for run, rows in sorted(groups.items()):
        complete = [r for r in rows if r["status"] == "complete"]
        summary = dict(run=run, profiles=len(rows), complete=len(complete), incomplete=len(rows)-len(complete))
        for key in ("policy", "budget", "duration_sec", "frames", "bank_device"):
            require(len({r[key] for r in rows}) == 1, f"Mixed group protocol: {run}/{key}")
            summary[key] = rows[0][key]
        if complete:
            for key in ("archive_frames_max", "archive_rgb_peak_mib", "archive_features_logical_peak_mib",
                        "host_rss_sampled_peak_gib", "torch_allocated_rollout_peak_gib",
                        "torch_reserved_rollout_peak_gib"):
                summary[key] = max(r[key] for r in complete)
            for key in ("rollout_s", "context_selection_total_s", "context_selection_mean_s_per_section",
                        "memory_policy_update_total_s", "memory_policy_update_mean_s_per_section",
                        "bank_store_total_s"):
                summary["mean_" + key] = mean(r[key] for r in complete)
        summaries.append(summary)
    elapsed, elapsed_summary = [], []
    for duration, root in ((60, "context_memory_60s"), (180, "context_180s")):
        for prefix in (("seed0_", "seed1_extra60s_") if duration == 60 else ("seed0_",)):
            matched = match_elapsed(statuses[root + "/baseline/run_status.jsonl"],
                                    statuses[root + "/slam_b32_covisibility/run_status.jsonl"], duration, prefix)
            elapsed.extend(matched)
            elapsed_summary.append(dict(duration_sec=duration, cohort_prefix=prefix, trajectories=len(matched),
                frames=matched[0]["frames"], steps=50,
                unbounded_mean_elapsed_min=mean(r["unbounded_elapsed_s"] for r in matched)/60,
                keepsake_b32_mean_elapsed_min=mean(r["keepsake_b32_elapsed_s"] for r in matched)/60,
                hardware_matching="unverified"))
    output.mkdir(parents=True)
    write_csv(output / "profiles_per_video.csv", profiles)
    write_csv(output / "profile_summary.csv", summaries)
    write_csv(output / "identity_matched_elapsed.csv", elapsed)
    write_csv(output / "elapsed_summary.csv", elapsed_summary)
    with bundle.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    (output / "provenance.json").write_text(json.dumps(dict(bundle=str(bundle), sha256=digest, members=hashes), indent=2)+"\n")
    status = dict(profiles=len(profiles), complete_profiles=sum(r["status"] == "complete" for r in profiles),
                  profile_groups=len(groups), matched_hardware_verified=False,
                  descriptor_only_timer_available=False, new_gpu_runs=0)
    (output / "status.json").write_text(json.dumps(status, indent=2)+"\n")
    lines = [r"\begin{table}[ht]", r"\centering", r"\small", r"\begin{tabular}{lrrr}",
             r"\toprule", r"Horizon & Matched trajectories & Unbounded (min) & KEEPSAKE-B32 (min) \\", r"\midrule"]
    for row in elapsed_summary:
        if row["cohort_prefix"] == "seed0_":
            lines.append(f"{row['duration_sec']} s & {row['trajectories']} & {row['unbounded_mean_elapsed_min']:.2f} & {row['keepsake_b32_mean_elapsed_min']:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Mean logged per-video elapsed times for the original fifteen output identities at each horizon (50 denoising steps). Hardware and software equivalence are not recorded in these logs; these are historical observations, not a controlled speedup benchmark. The timer is not retrieval-only latency.}",
              r"\label{tab:historical-elapsed}", r"\end{table}"]
    (output / "elapsed.tex").write_text("\n".join(lines)+"\n")
    chosen = {
        "context_memory_attention_30s/unbounded_attention_audit": "Unbounded (attention audit)",
        "context_memory_60s/slam_b128_covisibility": "KEEPSAKE-B128",
        "context_memory_60s/kcenter_b32": "$k$-center B32",
        "context_memory_60s/mce_b32_lambda1_pilot": "MCE B32",
    }
    lines = [r"\begin{table}[ht]", r"\centering", r"\small", r"\begin{tabular}{lrrrrrr}",
             r"\toprule", r"Profile & Horizon & $n$ & RGB (MiB) & RSS (GiB) & Alloc. (GiB) & Res. (GiB) \\", r"\midrule"]
    for row in summaries:
        if row["run"] in chosen:
            lines.append(f"{chosen[row['run']]} & {row['duration_sec']} s & {row['complete']} & " +
                         " & ".join(f"{row[k]:.3f}" for k in ("archive_rgb_peak_mib", "host_rss_sampled_peak_gib",
                                                            "torch_allocated_rollout_peak_gib", "torch_reserved_rollout_peak_gib")) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Separate historical profiles, all with CPU archives. Each memory entry is the maximum across completed runs: recorded logical RGB payload, sampled host RSS, and PyTorch rollout allocated/reserved peaks. These rows are not matched resource comparisons: horizons, budgets, cohorts and instrumentation differ, and hardware equivalence is unverified. Descriptor payload and container overhead are not included in RGB bytes.}",
              r"\label{tab:historical-profiles}", r"\end{table}"]
    (output / "profiles.tex").write_text("\n".join(lines)+"\n")
    print(json.dumps(status, indent=2))
    for row in elapsed_summary:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.bundle, args.output)
