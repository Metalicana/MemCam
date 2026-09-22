"""Aggregate matched, common-source retrieval records for the motivation panel."""

import csv
import hashlib

import numpy as np

from paper.plot_retrieval_deterioration import FIELDS, IDENTITY, load_sections, summarize


RUNS = (("baseline", "Unbounded"), ("slam_b32_covisibility", "KEEPSAKE (B32)"))


def compare(path, duration=60, expected_videos=15, fps=30, bins=8, bootstrap=10000, seed=0):
    if not np.isfinite(fps) or fps <= 0 or duration <= 0 or expected_videos < 2:
        raise ValueError("Need positive duration/FPS and at least two trajectories")
    queries = {run: set() for run, _ in RUNS}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {*IDENTITY, *FIELDS, "run_name", "content_run", "budget", "section_idx",
                    "target_frame", "selected_memory_frame", "candidate_count_mismatch"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing common-source retrieval columns")
        for row in reader:
            run = row["run_name"]
            if run not in queries or int(row["duration_sec"]) != duration:
                continue
            if row["content_run"] != "baseline":
                raise ValueError("Both selectors must use the same baseline source pixels")
            if row["budget"] != ("" if run == "baseline" else "32"):
                raise ValueError("Expected Unbounded and B32 selectors")
            values = np.asarray([float(row[m]) for m in FIELDS])
            if not np.isfinite(values).all() or (values < -1e-5).any() or (values > 2 + 1e-5).any():
                raise ValueError("Invalid DINO distance")
            section, target = int(row["section_idx"]), int(row["target_frame"])
            if section < 1 or not 0 <= int(row["selected_memory_frame"]) < target:
                raise ValueError("Selected memory must precede the target")
            if float(row["candidate_count_mismatch"]) != 0:
                raise ValueError("Candidate-bank reconstruction mismatch")
            key = (*[row[k] for k in IDENTITY], section, target)
            if key in queries[run]:
                raise ValueError("Duplicate retrieval query")
            queries[run].add(key)
    if queries[RUNS[0][0]] != queries[RUNS[1][0]]:
        raise ValueError("Selectors must cover exactly the same trajectories and queries")

    curves, arrays = [], []
    cohort = None
    for run, policy in RUNS:
        identities, sections, values, targets, count = load_sections(path, run, duration, expected_videos)
        if cohort is not None and cohort != (identities, sections):
            raise ValueError("Mismatched trajectory/section coverage")
        cohort = identities, sections
        result = summarize(values, targets, fps, bins, bootstrap, seed)
        arrays.append(values)
        for b, time in enumerate(result["time_sec"]):
            for m, metric in enumerate(FIELDS):
                curves.append(dict(run=run, policy=policy, metric=metric, time_sec=float(time),
                                   mean=float(result["curve_mean"][b, m]),
                                   ci_low=float(result["curve_ci"][0, b, m]),
                                   ci_high=float(result["curve_ci"][1, b, m]),
                                   trajectories=len(identities), queries_total=count))
    # The same trajectory draws pair policies, metrics and time bins.
    trajectory_means = np.stack([values.mean(axis=1) for values in arrays])
    differences = trajectory_means[1] - trajectory_means[0]
    draws = np.random.default_rng(seed).integers(0, len(identities), (bootstrap, len(identities)))
    ci = np.quantile(differences[draws].mean(axis=1), [.025, .975], axis=0)
    summaries = [dict(metric=metric, unbounded=float(trajectory_means[0, :, m].mean()),
                      keepsake=float(trajectory_means[1, :, m].mean()),
                      paired_difference=float(differences[:, m].mean()),
                      paired_ci_low=float(ci[0, m]), paired_ci_high=float(ci[1, m]))
                 for m, metric in enumerate(FIELDS)]
    metadata = dict(source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    parameters=dict(duration=duration, expected_videos=expected_videos, fps=fps,
                                    bins=bins, bootstrap=bootstrap, seed=seed, content_run="baseline"),
                    trajectories=[dict(zip(IDENTITY, i)) for i in identities], section_indices=sections,
                    queries_per_policy=len(queries[RUNS[0][0]]), summaries=summaries,
                    displayed_metric="selected_memory_corruption",
                    definition="1 - cosine(DINO(baseline image at selected historical index), "
                    "DINO(GT at that same historical index)); lower is better.",
                    method="Actual policy-selected IDs; fixed baseline historical pixels. Query means within "
                    "sections, equal section weights within each ordered-section bin, equal trajectory weights. "
                    "Shared trajectory-bootstrap draws, pointwise 95% percentile intervals.",
                    limitations="Not each policy's own generated memory pixels, not target-view matching, "
                    "and not a causal replay. Banks and choices come from each policy's own trace. "
                    "All initial-frame selections are included. View and effective mismatch are exported too. "
                    "CSV checks do not independently validate upstream features, video hashes or GT mapping.")
    return curves, metadata
