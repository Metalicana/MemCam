"""Paired scene-level uncertainty for existing, matched MemCam results."""

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))
from paper.finish_keepsake_pose_appearance import NORMALIZATION
from run_budget_metric_grid import quality_result, validate_bench, digest
from audit_metric_coverage import bench_details
from compare_fvd_matched import frechet_low_rank

RUNS = ("baseline", "fifo_b32", "mce_b32_lambda1_pilot", "kcenter_b32",
        "ri_b32_dino_rgb", "slam_b32_covisibility")
KEEP = RUNS[-1]


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p-values")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()


def mean_contrasts(values, draws=5000, seed=17, reference=KEEP):
    """Exact paired sign randomization; bootstrap whole scenes, not frames."""
    if draws < 1 or reference not in values or len(values) < 2:
        raise ValueError("Missing reference, comparisons or bootstrap draws")
    arrays = {run: np.asarray(a, dtype=float) for run, a in values.items()}
    shapes = {a.shape for a in arrays.values()}
    if len(shapes) != 1 or any(a.ndim != 1 or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Unpaired/nonfinite scene values")
    n = len(arrays[reference])
    if not 2 <= n <= 20:
        raise ValueError("Exact sign test supports 2..20 independent scenes")
    indices = np.random.default_rng(seed).integers(n, size=(draws, n))
    summaries, comparisons = [], []
    for run, a in arrays.items():
        low, high = np.percentile(a[indices].mean(1), [2.5, 97.5])
        summaries.append(dict(run=run, videos=n, mean=float(a.mean()), ci_low=float(low), ci_high=float(high)))
        if run == reference:
            continue
        delta = arrays[reference] - a
        low, high = np.percentile(delta[indices].mean(1), [2.5, 97.5])
        exceed = 0
        for first in range(0, 2**n, 2048):
            masks = np.arange(first, min(first + 2048, 2**n), dtype=np.uint64)
            signs = 2 * ((masks[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(float) - 1
            exceed += np.count_nonzero(np.abs(signs @ delta / n) >= abs(delta.mean()) - 1e-12)
        comparisons.append(dict(reference=reference, compared=run, videos=n,
                                difference=float(delta.mean()), ci_low=float(low), ci_high=float(high),
                                p_two_sided=float(exceed / 2**n)))
    return summaries, comparisons


def scalar_inputs(scores_path, items):
    """Read the raw files selected by the existing metric report, never its means alone."""
    names = [i["output_prefix"] + "custom.mp4" for i in items]
    if len(items) != 15 or len(set(names)) != 15 or len({i["scene"] for i in items}) != 15:
        raise ValueError("Requires the fixed 15 distinct scenes, one rollout per scene")
    with Path(scores_path).open() as handle:
        rows = list(csv.DictReader(handle))
    selected = {}
    for r in rows:
        if r["run"] not in RUNS or r["evaluator"] not in ("quality", "vbench"):
            continue
        key = (r["run"], r["evaluator"], r["metric"])
        if key in selected or int(r["n"]) != len(items):
            raise ValueError("Duplicate or wrong-size reported metric")
        selected[key] = r
    values = {metric: {} for metric in ("LPIPS", *NORMALIZATION, "custom_vbench6_percent")}
    sources = {str(scores_path): digest(scores_path)}
    per_video = []
    for run in RUNS:
        row = selected[(run, "quality", "LPIPS")]
        path = Path(row["source"])
        checked = quality_result(path, "LPIPS", run, names, 1825)
        if not np.isclose(checked["value"], float(row["value"]), atol=1e-10, rtol=0):
            raise ValueError("LPIPS no longer matches the selected report")
        raw = [json.loads(line) for line in (path.parent / "metrics.jsonl").read_text().splitlines() if line.strip()]
        raw = [r for r in raw if int(r["duration_sec"]) == 60]
        by_name = {Path(r["output"]).name: r for r in raw}
        for name, item in zip(names, items):
            r = by_name[name]
            if r["scene"] != item["scene"] or int(r["start_frame"]) != int(item["start_frame"]):
                raise ValueError("LPIPS dataset identity mismatch")
        values["LPIPS"][run] = np.array([float(by_name[n]["lpips_alex"]) for n in names])
        sources.update({str(path.parent / name): digest(path.parent / name)
                        for name in ("metrics.jsonl", "summary.json")})
        vb_paths = {selected[(run, "vbench", dim)]["source"] for dim in NORMALIZATION}
        if len(vb_paths) != 1:
            raise ValueError("Expected one VBench file containing all six dimensions")
        vb_path = Path(vb_paths.pop())
        checked = validate_bench(vb_path.parent, names, run)
        payload = json.loads(vb_path.read_text())
        for result in checked:
            dim = result["metric"]
            if not np.isclose(result["value"], float(selected[(run, "vbench", dim)]["value"]), atol=1e-10, rtol=0):
                raise ValueError(f"{run}/{dim} no longer matches selected report")
            by_name = {Path(r["video_path"]).name: r for r in bench_details(payload[dim])}
            scale = 100 if dim == "imaging_quality" else 1
            values[dim][run] = np.array([float(by_name[n]["video_results"]) / scale for n in names])
        sources[str(vb_path)] = digest(vb_path)
        values["custom_vbench6_percent"][run] = 100 * sum(
            w * (values[k][run] - lo) / (hi - lo) for k, (lo, hi, w) in NORMALIZATION.items()) / 5.5
        for index, item in enumerate(items):
            per_video.append(dict(run=run, scene=item["scene"], start_frame=item["start_frame"],
                                  output_prefix=item["output_prefix"], duration_sec=60,
                                  **{metric: float(a[run][index]) for metric, a in values.items()}))
    return values, sources, per_video


def export_scalars(scores_path, items, output, draws=5000):
    values, sources, per_video = scalar_inputs(scores_path, items)
    summaries, contrasts = [], []
    for metric, arrays in values.items():
        a, b = mean_contrasts(arrays, draws=draws)
        summaries.extend(dict(metric=metric, **r) for r in a)
        contrasts.extend(dict(metric=metric, **r) for r in b)
    for r, p in zip(contrasts, holm([r["p_two_sided"] for r in contrasts])):
        r["p_holm_scalar_family"] = p
    write_csv(output / "per_video.csv", per_video)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "contrasts.csv", contrasts)
    return dict(sources=sources, draws=draws, seed=17, normalization=NORMALIZATION,
                uncertainty="Whole-scene paired percentile bootstrap; one recorded seed per trajectory. Not seed variance.",
                hypothesis_test="Exact paired sign randomization under within-scene label exchangeability, two-sided.",
                family="All 40 KEEPSAKE contrasts across five comparators and eight scalar metrics; Holm FWER.",
                scope="Equal-video means, matching the metric grid; custom normalized six-score aggregate, not official full VBench.",
                provenance_limit="Recorded cohort/configuration validated; historical video hashes are not certified by legacy metric files.")


def fvd_statistics(features, draws=2000, seed=17, runs=RUNS):
    if set(features) != {"GT", *runs} or KEEP not in runs or len(set(runs)) != len(runs) or draws < 1:
        raise ValueError("Missing full policy FVD grid")
    shapes = {a.shape for a in features.values()}
    if len(shapes) != 1 or any(a.ndim != 3 or not np.isfinite(a).all() for a in features.values()):
        raise ValueError("Nonfinite or unpaired FVD features")
    n = features["GT"].shape[0]
    if n < 2:
        raise ValueError("Need multiple scenes")
    flat = lambda a: a.reshape(-1, a.shape[-1])
    score = lambda gt, gen: frechet_low_rank(flat(gt), flat(gen))
    estimates = {r: score(features["GT"], features[r]) for r in runs}
    boot = np.empty((draws, len(runs)))
    rng = np.random.default_rng(seed)
    for b in range(draws):
        ix = rng.integers(n, size=n)
        for j, run in enumerate(runs):
            boot[b, j] = score(features["GT"][ix], features[run][ix])
    summaries, contrasts = [], []
    for j, run in enumerate(runs):
        low, high = np.percentile(boot[:, j], [2.5, 97.5])
        summaries.append(dict(run=run, videos=n, fvd=estimates[run], ci_low=float(low), ci_high=float(high)))
        if run == KEEP:
            continue
        low, high = np.percentile(boot[:, runs.index(KEEP)] - boot[:, j], [2.5, 97.5])
        observed = estimates[KEEP] - estimates[run]
        percent = 100 * observed / estimates[run] if estimates[run] > 1e-10 else None
        relative_ci = (None, None)
        if np.all(boot[:, j] > 1e-10):
            relative_ci = tuple(float(v) for v in np.percentile(
                100 * (boot[:, runs.index(KEEP)] - boot[:, j]) / boot[:, j], [2.5, 97.5]))
        exceed = 0
        for _ in range(draws):
            mask = rng.integers(2, size=(n, 1, 1)).astype(bool)
            a = np.where(mask, features[KEEP], features[run])
            b = np.where(mask, features[run], features[KEEP])
            null = score(features["GT"], a) - score(features["GT"], b)
            exceed += abs(null) >= abs(observed) - 1e-10
        contrasts.append(dict(reference=KEEP, compared=run, videos=n, difference=observed,
                              ci_low=float(low), ci_high=float(high), p_two_sided=(exceed + 1) / (draws + 1),
                              relative_difference_percent=percent, relative_ci_low=relative_ci[0],
                              relative_ci_high=relative_ci[1]))
    for row, p in zip(contrasts, holm([r["p_two_sided"] for r in contrasts])):
        row["p_holm_fvd_family"] = p
    return summaries, contrasts, boot
