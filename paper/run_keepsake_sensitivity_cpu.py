"""Resumable CPU-only tau/beta/lambda sensitivity on fixed, cached histories."""

import argparse
import fcntl
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

NAMES = ("default", "tau_1", "tau_5", "beta_0", "beta_1", "lam_0", "lam_0.5")
SCOPE = ("Fixed-history retention sensitivity on fifteen previously inspected 60-second "
         "MemCam trajectories, B32. GT is used only for offline scoring. No feature extraction, "
         "new generation, actual retrieval, LPIPS or FVD. Not a held-out robustness test.")


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def configurations():
    all_configs = replay.configurations()
    return {name: all_configs[name] for name in NAMES}


def check_hashes(hashes):
    for path, expected in hashes.items():
        if replay.sha256(Path(path)) != expected:
            raise ValueError(f"Changed source/artifact: {path}; do not mix experiments")


def code_files():
    paths = {Path(__file__).resolve(), ROOT / "diffsynth/pipelines/memory_policies.py",
             ROOT / "dataset/poses.py"}
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source:
            path = Path(source).resolve()
            if path.is_relative_to(ROOT) and path.suffix == ".py":
                paths.add(path)
    return {str(p): replay.sha256(p) for p in sorted(paths)}


def prepare(args, items):
    if len(items) != 15 or len({i["scene"] for i in items}) != 15:
        raise ValueError("Expected all fifteen distinct matched scene IDs")
    sidecars, encoder = {}, None
    for item in items:
        if Path(item["output_prefix"]).name != item["output_prefix"] or int(item["start_frame"]) < 0:
            raise ValueError("Invalid trajectory identity")
        for kind in ("baseline", "gt"):
            path = args.cache / kind / (item["output_prefix"] + "dino.json")
            meta = json.loads(path.read_text())
            expected = dict(scene=item["scene"], dataset_start_frame=item["start_frame"],
                            duration_sec=60, num_frames=1825, output_prefix=item["output_prefix"], kind=kind)
            if any(meta.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Cache identity mismatch: {path}")
            if not meta.get("encoder") or (encoder is not None and encoder != meta["encoder"]):
                raise ValueError("All trajectories and GT/baseline caches must share the encoder identity")
            encoder = meta["encoder"]
            if kind == "gt" and meta.get("mode") != "fresh":
                raise ValueError("Fresh GT caches required")
            if not path.with_suffix(".npy").is_file():
                raise FileNotFoundError(path.with_suffix(".npy"))
            sidecars[str(path)] = replay.sha256(path)
    code = code_files()
    plan = dict(schema=1, scope=SCOPE, configurations=configurations(), items=items,
                root=str(args.root), cache=str(args.cache), manifest=str(args.manifest),
                manifest_sha256=replay.sha256(args.manifest), sidecars=sidecars, encoder=encoder,
                code_hashes=code, bootstrap_draws=5000, bootstrap_seed=17,
                runtime=dict(python=platform.python_version(), numpy=np.__version__),
                sampling="Four targets per section, slots 0,19,38,57; equal trajectory weights",
                timing="CPU affinity construction plus retention, excluding encoding and generation; not retrieval latency")
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


def load_poses(item):
    path = Path(item["pose_path"])
    keys = sorted(map(int, json.loads(path.read_text())["CineCameraActor"]))
    start, count = int(item["start_frame"]), int(item["num_frames"])
    if keys[start:start + count] != list(range(start, start + count)):
        raise ValueError("Noncontiguous pose/dataset indices")
    module = replay.load_file_module("sensitivity_poses", ROOT / "dataset/poses.py")
    return module.load_c2ws_from_json(path, start_frame=start, num_frames=count)


def validate_cell(payload, item):
    sections = (int(item["num_frames"]) - 1) // 76
    expected_queries = {(name, 76 * s + slot + 1) for name in NAMES
                        for s in range(1, sections) for slot in (0, 19, 38, 57)}
    queries, updates, banks = (payload[k] for k in ("queries", "updates", "banks"))
    if (len(queries) != len(expected_queries)
            or {(q["setting"], q["target_frame"]) for q in queries} != expected_queries):
        raise ValueError("Missing/duplicate sensitivity query results")
    expected_updates = {(name, s) for name in NAMES for s in range(sections)}
    for records in (updates, banks):
        if (len(records) != len(expected_updates)
                or {(r["setting"], r["section_idx"]) for r in records} != expected_updates):
            raise ValueError("Missing/duplicate sensitivity update results")
    if any(r["scene"] != item["scene"] for rows in (queries, updates, banks) for r in rows):
        raise ValueError("Wrong scene in replay outputs")
    for row in queries:
        if (not all(np.isfinite(row[k]) for k in ("retention_gap", "bank_oracle_distance", "full_oracle_distance"))
                or row["retention_gap"] < -1e-5 or not 1 <= row["retained_count"] <= 32):
            raise ValueError("Invalid retention scores")
    for row in banks:
        ids = row["retained_ids"]
        endpoint = (row["section_idx"] + 1) * 76
        if len(ids) != 32 or len(set(ids)) != 32 or min(ids) < 0 or max(ids) > endpoint or not {0, endpoint} <= set(ids):
            raise ValueError("Invalid causal B32 bank or protection")


def run_cell(args, item, policies, plan):
    folder = args.output / "cells" / f"row_{item['_row']:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path, data_path = folder / "receipt.json", folder / "replay.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("item") != item or receipt.get("plan_sha256") != replay.sha256(args.output / "plan.json"):
            raise ValueError("Cell receipt does not match the frozen plan")
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
    pose_path = Path(item["pose_path"])
    sources[str(pose_path)] = replay.sha256(pose_path)
    poses = load_poses(item)
    print(f"REPLAY {item['scene']}: seven settings", flush=True)
    queries, updates, banks = replay.replay_item(item, poses, arrays["baseline"], arrays["gt"],
                                               policies, configs=plan["configurations"], budget=32)
    payload = dict(queries=queries, updates=updates, banks=banks)
    validate_cell(payload, item)
    check_hashes(plan["code_hashes"])
    save(data_path, payload)
    save(receipt_path, dict(item=item, plan_sha256=replay.sha256(args.output / "plan.json"),
        sources=sources, artifacts={str(data_path): replay.sha256(data_path)},
        elapsed_sec=time.monotonic() - started, host=platform.node(), job=os.environ.get("SLURM_JOB_ID")))
    return payload


def report(output, items, payloads, plan):
    if len(items) != 15 or len(payloads) != 15:
        raise ValueError("No final sensitivity table until all fifteen trajectories are complete")
    per_trajectory = []
    for item, payload in zip(items, payloads):
        validate_cell(payload, item)
        default_banks = {r["section_idx"]: set(r["retained_ids"]) for r in payload["banks"] if r["setting"] == "default"}
        for name in NAMES:
            q = [r for r in payload["queries"] if r["setting"] == name]
            u = [r for r in payload["updates"] if r["setting"] == name]
            banks = [r for r in payload["banks"] if r["setting"] == name]
            jaccard = [len(set(r["retained_ids"]) & default_banks[r["section_idx"]]) /
                       len(set(r["retained_ids"]) | default_banks[r["section_idx"]]) for r in banks]
            per_trajectory.append(dict(setting=name, scene=item["scene"], source_row=item["_row"],
                queries=len(q), bank_jaccard_vs_default=float(np.mean(jaccard)),
                **{k: float(np.mean([r[k] for r in q])) for k in ("retention_gap", "bank_oracle_distance", "full_oracle_distance")},
                **{k: float(np.mean([r[k] for r in u])) for k in ("update_ms", "edge_density", "isolated_fraction")}))
    values = {name: [r["retention_gap"] for r in per_trajectory if r["setting"] == name] for name in NAMES}
    summaries, contrasts = mean_contrasts(values, reference="default", draws=plan["bootstrap_draws"], seed=plan["bootstrap_seed"])
    for row in contrasts:
        row.pop("p_two_sided")
    differences = {r["compared"]: r for r in contrasts}
    rows = []
    for summary in summaries:
        name = summary["run"]
        config = plan["configurations"][name]
        delta = differences.get(name, dict(difference=0., ci_low=0., ci_high=0.))
        records = [r for r in per_trajectory if r["setting"] == name]
        rows.append(dict(setting=name, tau=config["tau"], beta=config["beta"], lambda_weight=config["lam"],
            videos=15, retention_gap=summary["mean"], ci_low=summary["ci_low"], ci_high=summary["ci_high"],
            variant_minus_default=-delta["difference"], difference_ci_low=-delta["ci_high"], difference_ci_high=-delta["ci_low"],
            **{k: float(np.mean([r[k] for r in records])) for k in
               ("bank_oracle_distance", "bank_jaccard_vs_default", "edge_density", "isolated_fraction", "update_ms")}))
    write_csv(output / "per_trajectory.csv", per_trajectory)
    write_csv(output / "summary.csv", rows)
    write_csv(output / "paired_default_minus_variant.csv", contrasts)
    for key, filename in (("queries", "query_retention.csv"), ("updates", "updates.csv")):
        write_csv(output / filename, [r for payload in payloads for r in payload[key]])
    lines = [r"% Requires booktabs. Offline retention diagnostic, not generated-video quality.",
             r"% 15 trajectories; 60 s; B32; 5,000 paired trajectory-bootstrap draws.",
             r"% Delta = variant minus default. Lower retention gap is better; no automatic significance claim.",
             r"\begin{tabular}{rrrrr}", r"\toprule",
             r"$\tau$ & $\beta$ & $\lambda$ & Retention gap [95\% CI] & $\Delta$ [95\% CI] \\", r"\midrule"]
    for row in rows:
        lines.append(f"{row['tau']:g} & {row['beta']:g} & {row['lambda_weight']:g} & "
            f"{row['retention_gap']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}] & "
            f"{row['variant_minus_default']:+.4f} [{row['difference_ci_low']:+.4f}, {row['difference_ci_high']:+.4f}] " + r"\\")
    (output / "sensitivity.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    max_delta = max(abs(r["variant_minus_default"]) for r in rows)
    (output / "interpretation.txt").write_text(
        f"Across the prespecified one-at-a-time grid, mean retention-gap point estimates range from "
        f"{min(r['retention_gap'] for r in rows):.6f} to {max(r['retention_gap'] for r in rows):.6f}; "
        f"the largest absolute deviation from default is {max_delta:.6f} DINO cosine-distance units. "
        "This describes fixed-history retention sensitivity, not equivalence, held-out robustness or generated-video quality. "
        "Intervals resample scene IDs as whole trajectories; shared-environment dependence and generation-seed variability are not modeled.\n")
    return rows


def execute(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(status="running", job=os.environ.get("SLURM_JOB_ID"), completed_trajectories=0,
                      expected_trajectories=15, scope=SCOPE)
        save(args.output / "status.json", status)
        try:
            items = cohort(args.manifest, 60)
            plan = prepare(args, items)
            policies = replay.load_file_module("sensitivity_policies", ROOT / "diffsynth/pipelines/memory_policies.py")
            results = []
            for index, item in enumerate(items, 1):
                check_hashes(plan["code_hashes"])
                check_hashes(plan["sidecars"])
                check_hashes({str(args.manifest): plan["manifest_sha256"]})
                started = time.monotonic()
                print(f"TRAJECTORY {index}/15 {item['scene']}", flush=True)
                results.append(run_cell(args, item, policies, plan))
                status.update(completed_trajectories=index, last_scene=item["scene"],
                              last_validation_and_replay_sec=time.monotonic() - started)
                save(args.output / "status.json", status)
                print(f"COMPLETE {index}/15: {time.monotonic() - started:.1f} s", flush=True)
            check_hashes(plan["code_hashes"])
            check_hashes(plan["sidecars"])
            check_hashes({str(args.manifest): plan["manifest_sha256"]})
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
    parser.add_argument("--output", type=Path, default=Path.home() / "memcam_results/keepsake_sensitivity_cpu_60s_n15")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Use the CPU batch script; do not run the full replay on a login node")
    for key in ("manifest", "root", "cache", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.output.is_relative_to(args.root) or args.output.is_relative_to(args.cache):
        parser.error("Keep the sensitivity output separate from source videos and caches")
    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}; completed trajectory receipts are reusable")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    execute(args)


if __name__ == "__main__":
    main()
