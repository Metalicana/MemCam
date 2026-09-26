"""Post-hoc component ablation of retention on fixed cached histories; CPU only."""

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time

if __name__ == "__main__":
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper import replay_keepsake_updates as replay
from paper.evidence_statistics import mean_contrasts, write_csv
from paper.finish_keepsake_pose_appearance import cohort
from paper.run_keepsake_sensitivity_cpu import check_hashes, load_poses, save

STUDY = "keepsake_component_proxy_cpu_v1"
SCOPE = ("Post-hoc fixed-history component diagnostic on fifteen previously inspected "
         "60-second MemCam trajectories, B32. All variants see the same unbounded-generated "
         "history. GT is used only for evaluation. No actual reader, new generation, LPIPS "
         "or FVD. This does not replace the inconclusive closed-loop component ablation.")
LABELS = {"full": "KEEPSAKE", "pose_only": "Without appearance", "appearance_only": "Without pose",
          "degree_only": "Without closest-substitute term", "closest_only": "Without degree term"}
NAMES = tuple(LABELS)
SLOTS = (0, 19, 38, 57)
SOURCE_FILES = (
    "paper/run_keepsake_component_proxy_cpu.py", "paper/run_keepsake_sensitivity_cpu.py",
    "paper/replay_keepsake_updates.py", "paper/audit_gap_inputs.py",
    "paper/benchmark_query_latency.py", "paper/cache_gap_source_features.py",
    "paper/evidence_statistics.py", "paper/finish_keepsake_pose_appearance.py",
    "utils/analyze_retrieval_quality_decomposition.py",
    "diffsynth/pipelines/memory_policies.py", "dataset/poses.py",
)


def configurations():
    result = {name: dict(geometry_weight=.65, visual_weight=.35, priority_mode="full",
                        n_other_observers=3, covisibility_threshold=.65) for name in NAMES}
    result["pose_only"].update(geometry_weight=1., visual_weight=0.)
    result["appearance_only"].update(geometry_weight=0., visual_weight=1.)
    result["degree_only"]["priority_mode"] = "degree_only"
    result["closest_only"]["priority_mode"] = "closest_only"
    return result


def prepare(args, items):
    if len(items) != 15 or len({i["scene"] for i in items}) != 15:
        raise ValueError("Expected all fifteen distinct matched scene IDs")
    sidecars, encoder = {}, None
    for item in items:
        prefix = item["output_prefix"]
        if not prefix or Path(prefix).name != prefix or int(item["start_frame"]) < 0:
            raise ValueError("Invalid trajectory identity")
        for kind in ("baseline", "gt"):
            path = args.cache / kind / (prefix + "dino.json")
            meta = json.loads(path.read_text())
            expected = dict(scene=item["scene"], dataset_start_frame=item["start_frame"],
                            duration_sec=60, num_frames=1825, output_prefix=prefix, kind=kind)
            if any(meta.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Cache identity mismatch: {path}")
            if not meta.get("encoder") or (encoder is not None and encoder != meta["encoder"]):
                raise ValueError("All baseline/GT caches must share the encoder identity")
            encoder = meta["encoder"]
            if kind == "gt" and meta.get("mode") != "fresh":
                raise ValueError("Fresh GT caches required")
            if not path.with_suffix(".npy").is_file():
                raise FileNotFoundError(path.with_suffix(".npy"))
            sidecars[str(path)] = replay.sha256(path)
    code = {str(ROOT / name): replay.sha256(ROOT / name) for name in SOURCE_FILES}
    plan = dict(study=STUDY, scope=SCOPE, configurations=configurations(), items=items,
                root=str(args.root), cache=str(args.cache), manifest=str(args.manifest),
                manifest_sha256=replay.sha256(args.manifest), sidecars=sidecars, encoder=encoder,
                pose_hashes={i["pose_path"]: replay.sha256(Path(i["pose_path"])) for i in items},
                code_hashes=code, budget=32, bootstrap_draws=5000, bootstrap_seed=17,
                runtime=dict(python=platform.python_version(), numpy=np.__version__,
                             torch=importlib.metadata.version("torch")),
                primary="Mean retained-bank oracle DINO distance minus full-history oracle distance; lower is better",
                sampling="92 targets/trajectory: slots 0,19,38,57 in sections 1..23; equal trajectory weights",
                protection="Initial frame and current endpoint both count in B32; former endpoints become evictable",
                feature_scope="Shared-source cached descriptors, not asserted byte-identical to online descriptors",
                inference="Post-hoc descriptive paired trajectory bootstrap; no confirmatory p-values, equivalence or quality claim")
    path = args.output / "plan.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("Frozen plan/code/runtime changed; use a new output directory")
    else:
        for source in code:
            target = args.output / "code" / Path(source).relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        save(path, plan)
    return plan


def replay_item(item, poses, features, gt, policies):
    frames = int(item["num_frames"])
    if frames != 1825 or poses.shape != (frames, 4, 4) or not np.isfinite(poses).all():
        raise ValueError("Invalid 60-second pose coverage")
    if (features.ndim != 2 or features.shape != gt.shape or features.shape[0] != frames
            or features.shape[1] < 1 or not np.isfinite(features).all() or not np.isfinite(gt).all()
            or any(not np.allclose(np.linalg.norm(a, axis=1), 1., atol=.002, rtol=0) for a in (features, gt))):
        raise ValueError("Expected aligned finite unit-normalized baseline/GT features")
    configs = configurations()
    buffers = {name: policies.FrameMemoryBuffer("slam_covisibility", budget=32, pinned_frames={0})
               for name in NAMES}
    for buffer in buffers.values():
        buffer.add(0)
    updates, queries, banks = [], [], []
    for section in range(24):
        if section:
            cutoff = section * 76 - 3
            targets = [section * 76 + slot + 1 for slot in SLOTS]
            distances = 1 - np.clip(gt[targets] @ features[:cutoff].T, -1, 1)
            full = distances.min(1)
            for name, buffer in buffers.items():
                eligible = [i for i in buffer.candidates() if i < cutoff]
                if not eligible:
                    raise ValueError("No eligible retained candidates")
                best = distances[:, eligible].min(1)
                for index, target in enumerate(targets):
                    queries.append(dict(setting=name, scene=item["scene"], source_row=item["_row"],
                        section_idx=section, target_frame=target, full_oracle_distance=float(full[index]),
                        bank_oracle_distance=float(best[index]), retention_gap=float(best[index] - full[index]),
                        retained_count=len(eligible)))
        for name, buffer in buffers.items():
            new_ids = list(range(section * 76, section * 76 + 77))
            bank = buffer.candidates()
            ids = bank + [i for i in new_ids if i not in bank]
            # Pass only currently available evidence, never future features or GT, to production scoring.
            current_features = {i: features[i] for i in ids}
            started = time.perf_counter()
            scores, details = policies.compute_slam_covisibility_scores(
                ids, poses[:new_ids[-1] + 1], pinned_frames={0}, dino_features=current_features,
                return_details=True, **configs[name])
            buffer.update(new_ids, eviction_scores=scores, protected_frames={new_ids[-1]})
            elapsed = (time.perf_counter() - started) * 1000
            degrees = np.array([details[i]["covisible_observers"] for i in ids])
            updates.append(dict(setting=name, scene=item["scene"], section_idx=section,
                candidate_count=len(ids), retained_count=len(buffer), update_ms=elapsed,
                edge_density=float(degrees.sum() / (len(ids) * (len(ids) - 1))),
                isolated_fraction=float(np.mean(degrees == 0))))
            banks.append(dict(setting=name, scene=item["scene"], section_idx=section,
                              retained_ids=buffer.candidates()))
    return dict(queries=queries, updates=updates, banks=banks)


def validate_cell(payload, item):
    queries, updates, banks = (payload[k] for k in ("queries", "updates", "banks"))
    expected = {(name, s * 76 + slot + 1) for name in NAMES for s in range(1, 24) for slot in SLOTS}
    if len(queries) != len(expected) or {(r["setting"], r["target_frame"]) for r in queries} != expected:
        raise ValueError("Missing/duplicate component query results")
    expected = {(name, s) for name in NAMES for s in range(24)}
    for records in (updates, banks):
        if len(records) != len(expected) or {(r["setting"], r["section_idx"]) for r in records} != expected:
            raise ValueError("Missing/duplicate component update results")
    if any(r["scene"] != item["scene"] for rows in (queries, updates, banks) for r in rows):
        raise ValueError("Wrong scene in replay outputs")
    bank_map = {(r["setting"], r["section_idx"]): r["retained_ids"] for r in banks}
    update_map = {(r["setting"], r["section_idx"]): r for r in updates}
    for (name, section), ids in bank_map.items():
        endpoint = (section + 1) * 76
        previous = bank_map[name, section - 1] if section else [0]
        candidates = set(previous) | set(range(section * 76, endpoint + 1))
        if (len(ids) != 32 or len(set(ids)) != 32 or any(type(i) is not int for i in ids)
                or not set(ids) <= candidates or not {0, endpoint} <= set(ids)):
            raise ValueError("Invalid causal B32 bank/protection or resurrected evidence")
        row = update_map[name, section]
        if (row["candidate_count"] != len(candidates) or row["retained_count"] != 32
                or not np.isfinite(row["update_ms"]) or row["update_ms"] < 0
                or any(not np.isfinite(row[k]) or not 0 <= row[k] <= 1
                       for k in ("edge_density", "isolated_fraction"))):
            raise ValueError("Invalid update diagnostics")
    oracles = {}
    for row in queries:
        section = (row["target_frame"] - 1) // 76
        eligible = [i for i in bank_map[row["setting"], section - 1] if i < section * 76 - 3]
        bank, full, gap = (row[k] for k in ("bank_oracle_distance", "full_oracle_distance", "retention_gap"))
        if (row["source_row"] != item["_row"] or row["section_idx"] != section
                or row["retained_count"] != len(eligible) or not 1 <= len(eligible) <= 32
                or not all(np.isfinite(v) and -1e-6 <= v <= 2 + 1e-6 for v in (bank, full, gap))
                or not np.isclose(gap, bank - full, atol=1e-6, rtol=0)):
            raise ValueError("Invalid retention scores or query identity")
        target = row["target_frame"]
        if target in oracles and not np.isclose(full, oracles[target], atol=1e-7, rtol=0):
            raise ValueError("Variants do not share the same full-history oracle")
        oracles[target] = full


def run_cell(args, item, policies, plan):
    folder = args.output / "cells" / f"row_{item['_row']:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path, data_path = folder / "receipt.json", folder / "replay.json"
    plan_hash = replay.sha256(args.output / "plan.json")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("item") != item or receipt.get("plan_sha256") != plan_hash
                or set(receipt.get("artifacts", {})) != {str(data_path)}):
            raise ValueError("Cell receipt does not match the frozen plan/artifact")
        check_hashes(receipt["sources"])
        check_hashes(receipt["artifacts"])
        payload = json.loads(data_path.read_text())
        validate_cell(payload, item)
        print(f"REUSE {item['scene']}", flush=True)
        return payload
    started = time.monotonic()
    print(f"VERIFY CACHE/SOURCES {item['scene']}", flush=True)
    arrays, sources = replay.verified_cache(args.cache, item, args.root)
    trace = args.root / "baseline/access_traces" / (item["output_prefix"] + "custom.jsonl")
    sources[str(trace)] = replay.audit_trace(trace, item, "unbounded", None)["sha256"]
    sources[item["pose_path"]] = plan["pose_hashes"][item["pose_path"]]
    poses = load_poses(item)
    print(f"REPLAY {item['scene']}: five components; production scorer/buffer", flush=True)
    payload = replay_item(item, poses, arrays["baseline"], arrays["gt"], policies)
    validate_cell(payload, item)
    check_hashes(sources)
    check_hashes(plan["code_hashes"])
    save(data_path, payload)
    save(receipt_path, dict(item=item, plan_sha256=plan_hash, sources=sources,
        artifacts={str(data_path): replay.sha256(data_path)}, elapsed_sec=time.monotonic() - started,
        host=platform.node(), job=os.environ.get("SLURM_JOB_ID")))
    return payload


def report(output, items, payloads, plan):
    if len(items) != 15 or len(payloads) != 15:
        raise ValueError("No final component table until all fifteen trajectories are complete")
    per_trajectory = []
    for item, payload in zip(items, payloads):
        validate_cell(payload, item)
        full_banks = {r["section_idx"]: set(r["retained_ids"]) for r in payload["banks"] if r["setting"] == "full"}
        for name in NAMES:
            queries = [r for r in payload["queries"] if r["setting"] == name]
            banks = [r for r in payload["banks"] if r["setting"] == name]
            overlap = [len(set(r["retained_ids"]) & full_banks[r["section_idx"]]) /
                       len(set(r["retained_ids"]) | full_banks[r["section_idx"]]) for r in banks]
            per_trajectory.append(dict(setting=name, scene=item["scene"], source_row=item["_row"],
                queries=len(queries), bank_jaccard_vs_full=float(np.mean(overlap)),
                **{k: float(np.mean([r[k] for r in queries])) for k in
                   ("retention_gap", "bank_oracle_distance", "full_oracle_distance")}))
    values = {name: [r["retention_gap"] for r in per_trajectory if r["setting"] == name] for name in NAMES}
    summaries, contrasts = mean_contrasts(values, reference="full", draws=plan["bootstrap_draws"], seed=plan["bootstrap_seed"])
    for row in contrasts:
        row.pop("p_two_sided")
    differences = {r["compared"]: r for r in contrasts}
    rows = []
    for summary in summaries:
        name = summary["run"]
        delta = differences.get(name, dict(difference=0., ci_low=0., ci_high=0.))
        records = [r for r in per_trajectory if r["setting"] == name]
        rows.append(dict(setting=name, label=LABELS[name], **plan["configurations"][name], trajectories=15,
            retention_gap=summary["mean"], ci_low=summary["ci_low"], ci_high=summary["ci_high"],
            variant_minus_full=-delta["difference"], difference_ci_low=-delta["ci_high"],
            difference_ci_high=-delta["ci_low"],
            **{k: float(np.mean([r[k] for r in records])) for k in
               ("bank_oracle_distance", "full_oracle_distance", "bank_jaccard_vs_full")}))
    write_csv(output / "per_trajectory.csv", per_trajectory)
    write_csv(output / "summary.csv", rows)
    write_csv(output / "paired_full_minus_variant.csv", contrasts)
    for key, filename in (("queries", "query_retention.csv"), ("updates", "updates.csv")):
        write_csv(output / filename, [r for payload in payloads for r in payload[key]])
    lines = [r"% Requires booktabs. Post-hoc fixed-history diagnostic, not generated-video quality.",
             r"% 15 trajectories, B32; 5,000 paired trajectory-bootstrap draws; descriptive 95% CIs.",
             r"% Delta = variant minus full. Lower gap is better. No equivalence/significance claim.",
             r"\begin{tabular}{lrrr}", r"\toprule",
             r"Setting & Retention gap [95\% CI] & $\Delta$ [95\% CI] & Bank Jaccard \\", r"\midrule"]
    for row in rows:
        lines.append(f"{row['label']} & {row['retention_gap']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}] & "
                     f"{row['variant_minus_full']:+.4f} [{row['difference_ci_low']:+.4f}, {row['difference_ci_high']:+.4f}] & "
                     f"{row['bank_jaccard_vs_full']:.3f} " + r"\\")
    (output / "component_proxy.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    (output / "interpretation.txt").write_text(SCOPE + "\n"
        "Positive variant-minus-full means the full bank loses less oracle evidence. Read the paired-difference "
        "intervals, not overlap of marginal intervals. These descriptive intervals are not multiplicity-adjusted. "
        "Zero-crossing intervals do not establish equivalence. The scoring metric remains DINO distance for all "
        "variants and is not independent of the appearance descriptor used by the policy. "
        "Inference resamples trajectory IDs, not queries; shared-environment dependence and seed variation are not modeled.\n")
    return rows


def validate_output(args):
    for source in (args.root, args.cache):
        if args.output.is_relative_to(source) or source.is_relative_to(args.output):
            raise ValueError("Keep proxy output separate from source videos and caches")
    if args.output.exists() and any(args.output.iterdir()):
        identity = args.output / "plan.json"
        if not identity.exists():
            identity = args.output / "status.json"
        if not identity.exists() or json.loads(identity.read_text()).get("study") != STUDY:
            raise ValueError("Refusing to overwrite another study or nonempty output directory")


def execute(args):
    validate_output(args)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(study=STUDY, status="running", job=os.environ.get("SLURM_JOB_ID"),
                      completed_trajectories=0, expected_trajectories=15, scope=SCOPE)
        save(args.output / "status.json", status)
        try:
            items = cohort(args.manifest, 60)
            plan = prepare(args, items)
            policies = replay.load_file_module("component_proxy_policies", ROOT / "diffsynth/pipelines/memory_policies.py")
            results = []
            for index, item in enumerate(items, 1):
                for hashes in (plan["code_hashes"], plan["sidecars"], plan["pose_hashes"],
                               {str(args.manifest): plan["manifest_sha256"]}):
                    check_hashes(hashes)
                started = time.monotonic()
                print(f"TRAJECTORY {index}/15 {item['scene']}", flush=True)
                results.append(run_cell(args, item, policies, plan))
                status.update(completed_trajectories=index, last_scene=item["scene"])
                save(args.output / "status.json", status)
                print(f"COMPLETE {index}/15: {time.monotonic() - started:.1f} s", flush=True)
            for hashes in (plan["code_hashes"], plan["sidecars"], plan["pose_hashes"],
                           {str(args.manifest): plan["manifest_sha256"]}):
                check_hashes(hashes)
            report(args.output, items, results, plan)
            status["status"] = "complete"
        except BaseException as exc:
            status.update(status="interrupted" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed",
                          error=str(exc))
            raise
        finally:
            save(args.output / "status.json", status)
        print(f"COMPLETE: {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--root", type=Path, default=Path.home() / "memcam_results/context_memory_60s")
    parser.add_argument("--cache", type=Path, default=Path.home() / "memcam_results/context_memory_60s/gap_feature_cache_fresh")
    parser.add_argument("--output", type=Path, default=Path.home() / "memcam_results/keepsake_component_proxy_cpu_60s_n15")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Use the CPU batch script; do not run the full replay on a login node")
    for key in ("manifest", "root", "cache", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}; completed trajectory receipts are reusable")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    execute(args)


if __name__ == "__main__":
    main()
