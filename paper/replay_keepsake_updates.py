"""CPU fixed-history sensitivity and update-rule replay, not new generation."""

import json
from pathlib import Path
import time

import numpy as np

from paper.audit_gap_inputs import audit_cache, audit_trace
from paper.benchmark_query_latency import load_file_module
from paper.cache_gap_source_features import sha256
from paper.evidence_statistics import write_csv, mean_contrasts

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = dict(alpha=.65, threshold=.65, tau=3., beta=.5, lam=.25, update="one_shot")


def configurations():
    result = {"default": dict(DEFAULT)}
    for key, choices in (("alpha", (0., .35, .5, .8, 1.)), ("threshold", (.5, .8)),
                         ("tau", (1., 5.)), ("beta", (0., 1.)), ("lam", (0., .5))):
        for value in choices:
            result[f"{key}_{value:g}"] = {**DEFAULT, key: value}
    for rule in ("iterative", "frozen_admission"):
        result[rule] = {**DEFAULT, "update": rule}
    return result


def priority(affinity, config):
    a = np.asarray(affinity, dtype=float)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or not len(a)
            or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1)
            or not np.allclose(a, a.T) or not np.allclose(np.diag(a), 0)):
        raise ValueError("Expected finite symmetric [0,1] affinity with zero diagonal")
    if not 0 < config["threshold"] <= 1 or config["tau"] < 1:
        raise ValueError("Invalid threshold/tau")
    count = (a >= config["threshold"]).sum(1)
    return 1 - np.minimum(count / config["tau"], 1) + config["beta"] / (count + 1) + config["lam"] * (1 - a.max(1))


def retain(ids, affinity, budget, protected, config, admission_scores):
    """Recompute graph statistics, not the pose normalization, within iterative deletion."""
    if (len(set(ids)) != len(ids) or not set(protected) <= set(ids)
            or budget < len(protected) or budget < 1 or len(ids) != len(affinity)):
        raise ValueError("Invalid IDs, protection or budget")
    scores = priority(affinity, config)
    if config["update"] == "frozen_admission":
        for frame, score in zip(ids, scores):
            admission_scores.setdefault(frame, float(score))
        scores = np.array([admission_scores[frame] for frame in ids])
    elif config["update"] not in ("one_shot", "iterative"):
        raise ValueError("Unknown update rule")
    active = list(range(len(ids)))
    while len(active) > budget:
        if config["update"] == "iterative":
            current = priority(affinity[np.ix_(active, active)], config)
            local = dict(zip(active, current))
        else:
            local = dict(enumerate(scores))
        # IDs arrive in insertion order; tie-breaking stays oldest first.
        victim = min((i for i in active if ids[i] not in protected), key=lambda i: (local[i], i))
        active.remove(victim)
    return [ids[i] for i in active]


def verified_cache(cache, item, root):
    arrays, sources, encoder = {}, {}, None
    expected = dict(scene=item["scene"], dataset_start_frame=item["start_frame"], duration_sec=60,
                    num_frames=item["num_frames"], output_prefix=item["output_prefix"])
    for kind in ("baseline", "gt"):
        path = cache / kind / (item["output_prefix"] + "dino.npy")
        meta = json.loads(path.with_suffix(".json").read_text())
        if any(meta.get(k) != v for k, v in expected.items()) or meta.get("kind") != kind:
            raise ValueError(f"Cache identity mismatch: {path}")
        if not meta.get("encoder") or (encoder is not None and encoder != meta["encoder"]):
            raise ValueError("Source/GT encoders differ or lack provenance")
        encoder = meta["encoder"]
        if meta["feature_sha256"] != sha256(path):
            raise ValueError(f"Corrupt feature cache: {path}")
        audit_cache(path, int(item["num_frames"]))
        if kind == "baseline":
            source = root / "baseline" / (item["output_prefix"] + "custom.mp4")
            if meta["source_sha256"] != sha256(source):
                raise ValueError("Baseline video differs from cached feature source")
            sources[str(source)] = meta["source_sha256"]
        else:
            if meta.get("mode") != "fresh":
                raise ValueError("Replay requires the freshly encoded GT cache")
            frames = meta["source_frames"]
            start = int(item["start_frame"])
            if [f["index"] for f in frames] != list(range(start, start + int(item["num_frames"]))):
                raise ValueError("Wrong GT frame map")
            for record in frames:
                source = Path(item["gt_frames_dir"]) / f"{record['index']:04d}.png"
                if sha256(source) != record["sha256"]:
                    raise ValueError(f"GT source changed: {source}")
                sources[str(source)] = record["sha256"]
        arrays[kind] = np.asarray(np.load(path, allow_pickle=False)[:int(item["num_frames"])], dtype=np.float32)
        sources[str(path)] = meta["feature_sha256"]
        sources[str(path.with_suffix('.json'))] = sha256(path.with_suffix('.json'))
    return arrays, sources


def replay_item(item, poses, features, gt, policies, configs=None, budget=32):
    configs = configs or configurations()
    frames = int(item["num_frames"])
    if (frames - 1) % 76 or poses.shape != (frames, 4, 4) or not np.isfinite(poses).all():
        raise ValueError("Invalid pose coverage or chunk alignment")
    banks = {name: [0] for name in configs}
    admission = {name: {} for name in configs}
    features_map = dict(enumerate(features))
    updates, queries, snapshots = [], [], []
    for section in range((frames - 1) // 76):
        if section:
            history = list(range(section * 76 - 3))
            targets = [section * 76 + slot + 1 for slot in (0, 19, 38, 57)]
            distances = 1 - np.clip(gt[targets] @ features[history].T, -1, 1)
            full = distances.min(1)
            for name, bank in banks.items():
                eligible = [i for i in bank if i < section * 76 - 3]
                if not eligible:
                    raise ValueError("No eligible retained candidates")
                best = distances[:, eligible].min(1)
                if np.any(best < full - 1e-5):
                    raise ValueError("Negative retention gap")
                for index, target in enumerate(targets):
                    queries.append(dict(setting=name, scene=item["scene"], source_row=item["_row"],
                        section_idx=section, target_frame=target, full_oracle_distance=float(full[index]),
                        bank_oracle_distance=float(best[index]), retention_gap=float(best[index]-full[index]),
                        retained_count=len(eligible)))
        for name, config in configs.items():
            bank = banks[name]
            new_ids = list(range(section * 76, section * 76 + 77))
            ids = bank + [i for i in new_ids if i not in bank]
            started = time.perf_counter()
            affinity = policies._slam_covisibility_affinity(ids, poses, dino_features=features_map,
                         geometry_weight=config["alpha"], visual_weight=1-config["alpha"])
            if name == "default" and section == 0:
                checked_at = time.perf_counter()
                production = policies.compute_slam_covisibility_scores(ids, poses, dino_features=features_map)
                if not np.allclose(priority(affinity, config), [production[i] for i in ids], atol=1e-10, rtol=0):
                    raise ValueError("Offline default does not reproduce production scores")
                started += time.perf_counter() - checked_at
            banks[name] = retain(ids, affinity, budget, {0, new_ids[-1]}, config, admission[name])
            elapsed = (time.perf_counter() - started) * 1000
            degrees = (affinity >= config["threshold"]).sum(1)
            updates.append(dict(setting=name, scene=item["scene"], section_idx=section,
                candidate_count=len(ids), retained_count=len(banks[name]), update_ms=elapsed,
                edge_density=float(degrees.sum() / max(1, len(ids)*(len(ids)-1))),
                isolated_fraction=float(np.mean(degrees == 0))))
            snapshots.append(dict(setting=name, section_idx=section, scene=item["scene"], retained_ids=banks[name]))
    return queries, updates, snapshots


def run_replay(items, root, cache, output):
    policies = load_file_module("replay_policies", ROOT / "diffsynth/pipelines/memory_policies.py")
    pose_module = load_file_module("replay_poses", ROOT / "dataset/poses.py")
    all_queries, all_updates, sources = [], [], {}
    for i, item in enumerate(items):
        print(f"Replay {i+1}/{len(items)}: {item['scene']}", flush=True)
        arrays, current_sources = verified_cache(cache, item, root)
        sources.update(current_sources)
        trace = root / "baseline/access_traces" / (item["output_prefix"] + "custom.jsonl")
        audit = audit_trace(trace, item, "unbounded", None)
        sources[str(trace)] = audit["sha256"]
        path = Path(item["pose_path"])
        keys = sorted(map(int, json.loads(path.read_text())["CineCameraActor"]))
        start, count = int(item["start_frame"]), int(item["num_frames"])
        if keys[start:start+count] != list(range(start, start+count)):
            raise ValueError("Noncontiguous pose/dataset indices")
        poses = pose_module.load_c2ws_from_json(path, start_frame=start, num_frames=count)
        sources[str(path)] = sha256(path)
        queries, updates, snapshots = replay_item(item, poses, arrays["baseline"], arrays["gt"], policies)
        all_queries.extend(queries)
        all_updates.extend(updates)
        (output / f"banks_row_{item['_row']:03d}.json").write_text(json.dumps(snapshots) + "\n")
    write_csv(output / "query_retention.csv", all_queries)
    write_csv(output / "updates.csv", all_updates)
    trajectories = []
    for name in configurations():
        for item in items:
            q = [r for r in all_queries if r["setting"] == name and r["scene"] == item["scene"]]
            u = [r for r in all_updates if r["setting"] == name and r["scene"] == item["scene"]]
            trajectories.append(dict(setting=name, scene=item["scene"], queries=len(q),
                **{k: float(np.mean([r[k] for r in q])) for k in ("retention_gap", "bank_oracle_distance", "full_oracle_distance")},
                **{k: float(np.mean([r[k] for r in u])) for k in ("update_ms", "edge_density", "isolated_fraction")}))
    write_csv(output / "trajectory_summary.csv", trajectories)
    values = {name: [r["retention_gap"] for r in trajectories if r["setting"] == name] for name in configurations()}
    summary, contrasts = mean_contrasts(values, reference="default")
    # This sweep is exploratory: export paired intervals without presenting a battery of significance claims.
    for r in contrasts:
        r.pop("p_two_sided")
    write_csv(output / "retention_summary.csv", summary)
    write_csv(output / "paired_default_minus_variant.csv", contrasts)
    return dict(sources=sources, configurations=configurations(), budget=32,
        scope="Fixed unbounded RGB/pose stream. Offline retention only; no actual selected reads, no selection gap, no generated-quality effect.",
        features="Cached DINOv2-base shared-source features, not claimed byte-identical to online descriptors.",
        protection="Initial frame and current endpoint; previous endpoints become evictable. Both count in B32.",
        iterative="Affinity and pose normalization fixed within each update; degrees/maxima recomputed after deletion.",
        frozen_admission="Finite priority at first insertion retained forever; protection masks eligibility, never freezes an infinite score.",
        timing="CPU affinity construction plus retention decision; single pass, includes diagnostics, excludes feature extraction and generation. Not retrieval latency.",
        sampling="Four fixed targets per retrieved section: slots 0,19,38,57. Equal trajectory weighting.",
        inference="Exploratory sensitivity on previously inspected scenes, not held-out tuning or a causal generation experiment.")
