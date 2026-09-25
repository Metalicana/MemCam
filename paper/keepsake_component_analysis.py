"""Prespecified, paired analysis for the full 60-second component study."""

import hashlib
import json
from pathlib import Path

import numpy as np

from paper import evidence_statistics as statistics
from paper.run_keepsake_component_pilot import SETTINGS, LABELS, METRICS
from utils.analyze_retrieval_quality_decomposition import reconstruct_candidate_banks

PROTOCOL = dict(
    duration_sec=60, frames=1825, fps=30, videos=15, budget=32, seed=42,
    steps=50, height=352, width=640, frame_stride=30,
    primary="Whole-rollout generated-frame LPIPS; four full-minus-removal contrasts, Holm correction",
    secondary="PSNR, SSIM, late/revisit fidelity, cohort FVD, and trace diagnostics",
    sampling="Frames 30,60,...,1800; initial conditioning frame excluded; equal trajectory weights",
    late_start_frame=1350, bootstrap_draws=5000, bootstrap_seed=17,
    fvd=dict(clips_per_video=4, clip_length=16, frame_stride=4, image_size=224,
             backend="styleganv_i3d", bootstrap_draws=2000),
    revisit=dict(min_gap_frames=450, position_m=.10, rotation_deg=5.,
                 excursion_m=.50, excursion_deg=30., consecutive_away_samples=3),
    scope="Follow-up on previously inspected scenes, not a new held-out test or seed-variance study",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(path, data):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError(f"Frozen input differs: {path}; use a new study directory")
    else:
        save(path, data)


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def pose_distances(poses, reference):
    position = np.linalg.norm(poses[:, :3, 3] - reference[:3, 3], axis=1)
    trace = np.einsum("nij,ij->n", poses[:, :3, :3], reference[:3, :3])
    angle = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
    return position, angle


def revisit_queries(poses):
    """Pose-only labels; stationary dwelling does not count as leaving and returning."""
    poses = np.asarray(poses, dtype=float)
    if poses.shape != (1825, 4, 4) or not np.isfinite(poses).all():
        raise ValueError("Expected all 1825 finite camera-to-world poses")
    rotations = poses[:, :3, :3]
    if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5)):
        raise ValueError("Invalid camera rotations")
    cfg = PROTOCOL["revisit"]
    samples = np.arange(30, 1801, 30)
    result = []
    for q in samples:
        history = samples[samples <= q - cfg["min_gap_frames"]]
        if not len(history):
            continue
        distances, angles = pose_distances(poses[history], poses[q])
        matches = history[(distances <= cfg["position_m"]) & (angles <= cfg["rotation_deg"])]
        # Earliest qualifying generated visit; no access to generated pixels or metrics.
        for first in matches:
            between = samples[(samples > first) & (samples < q)]
            d, a = pose_distances(poses[between], poses[first])
            away = (d > cfg["excursion_m"]) | (a > cfg["excursion_deg"])
            length = cfg["consecutive_away_samples"]
            if len(away) >= length and np.any(np.convolve(away.astype(int), np.ones(length), "valid") == length):
                result.append(dict(first_frame=int(first), target_frame=int(q)))
                break
    return result


def reduce_frames(rows, item, revisits):
    expected = list(range(0, 1825, 30))
    if [int(r["frame_index"]) for r in rows] != expected:
        raise ValueError("Incomplete, duplicate or unordered frame metrics")
    for row in rows:
        if (row["scene"] != item["scene"] or int(row["row"]) != item["_row"]
                or int(row["duration_sec"]) != 60
                or int(row["gt_frame_index"]) != item["start_frame"] + row["frame_index"]
                or any(not np.isfinite(float(row[k])) for k in METRICS)):
            raise ValueError("Mismatched or nonfinite frame metric")
    selected = {
        "all": set(range(30, 1801, 30)),
        "late": set(range(1380, 1801, 30)),
        "revisit": {r["target_frame"] for r in revisits},
    }
    result = []
    for window, ids in selected.items():
        subset = [r for r in rows if r["frame_index"] in ids]
        result.append(dict(window=window, sampled_frames=len(subset),
                           **{k: float(np.mean([r[k] for r in subset])) if subset else None for k in METRICS}))
    return result


def trace_signature(path):
    events = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    reads = {int(r["target_frame"]): r for r in events
             if r.get("event") == "context_access" and r.get("selected")}
    if set(reads) != set(range(77, 1825)):
        raise ValueError("Incomplete selected-read coverage")
    banks = reconstruct_candidate_banks(events, 23, num_frames=1825)
    return reads, banks


def trace_comparison(full_path, variant_path, poses, revisits):
    full, full_banks = trace_signature(full_path)
    variant, banks = trace_signature(variant_path)
    overlap = []
    for section in range(1, 24):
        left, right = set(full_banks[section]), set(banks[section])
        overlap.append(len(left & right) / len(left | right))
    available, selected = [], []
    cfg = PROTOCOL["revisit"]
    for r in revisits:
        q = r["target_frame"]
        read = variant[q]
        ids = np.asarray(banks[int(read["section_idx"])], dtype=int)
        d, a = pose_distances(poses[ids], poses[q])
        suitable = ids[(ids <= q - cfg["min_gap_frames"]) &
                       (d <= cfg["position_m"]) & (a <= cfg["rotation_deg"])]
        available.append(bool(len(suitable)))
        selected.append(int(read["selected_memory_frame"]) in suitable)
    return dict(bank_jaccard_vs_full=float(np.mean(overlap)),
                selected_id_agreement_vs_full=float(np.mean([
                    full[q]["selected_memory_frame"] == variant[q]["selected_memory_frame"] for q in full])),
                revisit_queries=len(revisits),
                revisit_old_view_available_fraction=float(np.mean(available)) if available else None,
                revisit_selected_old_view_fraction=float(np.mean(selected)) if selected else None)


def summarize(rows):
    expected = {(name, scene, window) for name, _, _ in SETTINGS
                for scene in range(15) for window in ("all", "late", "revisit")}
    by_key = {(r["setting"], r["scene_index"], r["window"]): r for r in rows}
    if len(rows) != len(by_key) or set(by_key) != expected:
        raise ValueError("Final analysis requires every setting and all 15 trajectories")
    summaries, contrasts = [], []
    for window in ("all", "late", "revisit"):
        scenes = [i for i in range(15) if by_key[("full", i, window)]["sampled_frames"] > 0]
        for i in range(15):
            counts = {by_key[(name, i, window)]["sampled_frames"] for name, _, _ in SETTINGS}
            if len(counts) != 1:
                raise ValueError("Unpaired temporal/query coverage")
        for metric in METRICS:
            values = {name: [by_key[(name, i, window)][metric] for i in scenes] for name, _, _ in SETTINGS}
            if len(scenes) >= 2:
                a, b = statistics.mean_contrasts(values, reference="full",
                    draws=PROTOCOL["bootstrap_draws"], seed=PROTOCOL["bootstrap_seed"])
            else:
                a = [dict(run=name, videos=len(scenes), mean=values[name][0] if scenes else None,
                          ci_low=None, ci_high=None) for name, _, _ in SETTINGS]
                b = []
            summaries.extend(dict(window=window, metric=metric, **r) for r in a)
            for r in b:
                r.update(window=window, metric=metric, p_holm_primary=None)
                if window != "all" or metric != "lpips_alex":
                    r["p_two_sided"] = None
                contrasts.append(r)
    primary = [r for r in contrasts if r["window"] == "all" and r["metric"] == "lpips_alex"]
    for row, p in zip(primary, statistics.holm([r["p_two_sided"] for r in primary])):
        row["p_holm_primary"] = p
    return summaries, contrasts


def summarize_fvd(features):
    names = [name for name, _, _ in SETTINGS]
    if set(features) != {"gt", *names}:
        raise ValueError("Missing paired FVD feature arrays")
    arrays = {name: np.asarray(value) for name, value in features.items()}
    if (len({a.shape for a in arrays.values()}) != 1
            or any(a.ndim != 3 or a.shape[:2] != (15, 4) or not np.isfinite(a).all() for a in arrays.values())):
        raise ValueError("FVD needs 15 paired scenes and four clips each")
    flat = lambda a: a.reshape(-1, a.shape[-1])
    score = lambda gt, gen: statistics.frechet_low_rank(flat(gt), flat(gen))
    estimates = {name: score(arrays["gt"], arrays[name]) for name in names}
    draws = PROTOCOL["fvd"]["bootstrap_draws"]
    rng = np.random.default_rng(PROTOCOL["bootstrap_seed"])
    boot = np.empty((draws, len(names)))
    for b in range(draws):
        ids = rng.integers(15, size=15)
        for j, name in enumerate(names):
            boot[b, j] = score(arrays["gt"][ids], arrays[name][ids])
    summaries, contrasts = [], []
    for j, name in enumerate(names):
        lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
        summaries.append(dict(setting=name, videos=15, clips=60, fvd=estimates[name], ci_low=float(lo), ci_high=float(hi)))
        if name != "full":
            lo, hi = np.percentile(boot[:, 0] - boot[:, j], [2.5, 97.5])
            contrasts.append(dict(reference="full", compared=name, videos=15,
                difference=estimates["full"]-estimates[name], ci_low=float(lo), ci_high=float(hi)))
    return summaries, contrasts
