"""Matched MemCam headline intervals from existing videos, never generation.

Keep the previously run paired extractor unchanged. Each comparison has its own
resumable extraction cache; inference uses joint whole-trajectory resampling.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
import zipfile

if __name__ == "__main__":
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper import paired_quality_uncertainty as pair
from paper.evidence_statistics import holm
from paper.finish_keepsake_pose_appearance import cohort

KEEP = "slam_b32_covisibility"
GROUPS = {
    "main_60s": (60, ("baseline", "fifo_b32", "mce_b32_lambda1_pilot", "kcenter_b32", KEEP)),
    "long_180s": (180, ("baseline", "fifo_b32", KEEP)),
    "budget_180s": (180, ("slam_b16_covisibility", KEEP, "slam_b64_covisibility", "slam_b128_covisibility")),
}
LABELS = {"baseline": "Unbounded", "fifo_b32": "FIFO B32",
          "mce_b32_lambda1_pilot": "MCE B32", "kcenter_b32": "K-center B32",
          **{f"slam_b{b}_covisibility": f"KEEPSAKE B{b}" for b in (16, 32, 64, 128)}}
# Manuscript points are an audit target, never substituted for recomputed values.
PUBLISHED = {
    "main_60s": {"baseline": (.5916, 784.9), "fifo_b32": (.6222, 832.2),
                 "mce_b32_lambda1_pilot": (.5985, 773.9), "kcenter_b32": (.5922, 712.0), KEEP: (.5850, 690.5)},
    "long_180s": {"baseline": (.5980, 734.2), "fifo_b32": (.6514, 677.3), KEEP: (.5876, 476.6)},
    "budget_180s": {"slam_b16_covisibility": (.5876, 446.1), KEEP: (.5876, 476.6),
                    "slam_b64_covisibility": (.5865, 493.9), "slam_b128_covisibility": (.5913, 515.5)},
}


def read(path):
    return json.loads(path.read_text())


def config(duration):
    return dict(clip_length=16, clips_per_video=4 if duration == 60 else 8,
                frame_stride=4 if duration == 60 else 8, image_size=224,
                backend="styleganv_i3d", eps=1e-6)


def validate_arrays(items, values, features):
    n = len(items)
    runs = tuple(values)
    if KEEP not in runs or len(runs) < 2 or set(features) != {"GT", *runs}:
        raise ValueError("Missing policies or common GT")
    if len({pair.identity(i) for i in items}) != n:
        raise ValueError("Duplicate trajectories")
    if any(np.asarray(a).shape != (n,) or not np.isfinite(a).all() for a in values.values()):
        raise ValueError("Unpaired/nonfinite scalar values")
    if (len({a.shape for a in features.values()}) != 1
            or any(a.ndim != 3 or a.shape[0] != n or min(a.shape[1:]) < 1
                   or not np.isfinite(a).all() for a in features.values())):
        raise ValueError("Unpaired/nonfinite clip features")


def statistics(items, values, features, draws=2000, swaps=5000, seed=17, mapping=None):
    """Unadjusted percentile CIs and paired label-swap tests; never per-video FVD."""
    values = {k: np.asarray(v, dtype=float) for k, v in values.items()}
    validate_arrays(items, values, features)
    if draws < 2 or swaps < 1:
        raise ValueError("Need at least two bootstrap draws and positive permutation draws")
    runs = tuple(values)
    labels, groups = pair.grouping(items, mapping)
    flat = lambda a: a.reshape(-1, a.shape[-1])
    score = lambda gt, gen: pair.frechet_low_rank(flat(gt), flat(gen))

    def evaluate(ix):
        return np.array([[np.mean(values[r][ix]), score(features["GT"][ix], features[r][ix])]
                         for r in runs])

    points = evaluate(np.arange(len(items)))
    boot = np.stack([evaluate(ix) for ix in pair.resamples(groups, draws, seed)])
    summaries, contrasts = [], []
    keep_index = runs.index(KEEP)
    for j, run in enumerate(runs):
        for k, metric in enumerate(("LPIPS", "FVD")):
            lo, hi = np.percentile(boot[:, j, k], [2.5, 97.5])
            summaries.append(dict(run=run, metric=metric, videos=len(items), groups=len(groups),
                                  estimate=float(points[j, k]), ci_low=float(lo), ci_high=float(hi)))
        if run == KEEP:
            continue
        observed = points[keep_index] - points[j]
        # Swap complete trajectories (all clips), or complete environment clusters.
        rng = np.random.default_rng(seed + 1)
        exact = len(groups) <= 20 and 2**len(groups) <= swaps
        count = 2**len(groups) if exact else swaps
        exceed = np.zeros(2, dtype=int)
        for b in range(count):
            bits = ((b >> np.arange(len(groups))) & 1) if exact else rng.integers(2, size=len(groups))
            mask = np.zeros(len(items), dtype=bool)
            for bit, group in zip(bits, groups):
                mask[group] = bool(bit)
            delta = values[KEEP] - values[run]
            null_lpips = np.mean(np.where(mask, -delta, delta))
            a = np.where(mask[:, None, None], features[run], features[KEEP])
            c = np.where(mask[:, None, None], features[KEEP], features[run])
            null_fvd = score(features["GT"], a) - score(features["GT"], c)
            exceed += np.abs([null_lpips, null_fvd]) >= np.abs(observed) - 1e-10
        pvalues = exceed/count if exact else (exceed+1)/(count+1)
        for k, metric in enumerate(("LPIPS", "FVD")):
            lo, hi = np.percentile(boot[:, keep_index, k] - boot[:, j, k], [2.5, 97.5])
            positive_denominator = points[j, k] > 1e-10 and np.all(boot[:, j, k] > 1e-10)
            rel = rlo = rhi = None
            if positive_denominator:
                rel = float(-100*observed[k]/points[j, k])
                rlo, rhi = map(float, np.percentile(
                    100*(boot[:, j, k]-boot[:, keep_index, k])/boot[:, j, k], [2.5, 97.5]))
            contrasts.append(dict(reference=KEEP, compared=run, metric=metric,
                videos=len(items), groups=len(groups), difference=float(observed[k]),
                ci_low=float(lo), ci_high=float(hi), reduction_percent=rel,
                reduction_ci_low=rlo, reduction_ci_high=rhi, p_two_sided=float(pvalues[k]),
                test="exact_cluster_label_swap" if exact else "monte_carlo_cluster_label_swap",
                permutations=count))
    for row, p in zip(contrasts, holm([r["p_two_sided"] for r in contrasts])):
        row["p_holm_group"] = p
    return summaries, contrasts, boot, labels


def pair_args(args, duration, run, output):
    source = args.root / ("context_memory_60s" if duration == 60 else "context_180s")
    return argparse.Namespace(output=output, duration=duration, expected_videos=15,
        baseline=source/run, keepsake=source/KEEP,
        manifest=ROOT/"testbeds"/("context_memory" if duration == 60 else "context_memory_180s")/"manifest.jsonl",
        detector=args.detector, lpips_stride=30 if duration == 60 else 90,
        clips=4 if duration == 60 else 8, clip_stride=4 if duration == 60 else 8)


def load_pair(directory, expected=None, check_sources=False):
    # Newton exposes the same home through /home and /lustre; receipts need one
    # canonical spelling so report-only resumption does not look like new data.
    directory = directory.resolve()
    plan_path = directory/"plan.json"
    plan = read(plan_path)
    if expected is not None:
        items, cfg = pair.load_protocol(expected)
        checks = dict(items=items, config=cfg, duration_sec=expected.duration, expected_videos=15,
                      lpips_stride=expected.lpips_stride, learned_image_size=224,
                      baseline=str(expected.baseline.resolve()), keepsake=str(expected.keepsake.resolve()),
                      detector_sha256=pair.digest(expected.detector), manifest_sha256=pair.digest(expected.manifest))
        if any(plan.get(k) != v for k, v in checks.items()):
            raise ValueError(f"Pair protocol mismatch: {directory}")
    values = {k: [] for k in pair.LABELS}
    features = {k: [] for k in ("GT", *pair.LABELS)}
    encoder = None
    artifacts = {str(plan_path): pair.digest(plan_path)}
    for index, item in enumerate(plan["items"]):
        path = directory/"cells"/f"pair_{index:03d}.json"
        row = read(path)
        if row["item"] != item or row["inputs"]["plan_sha256"] != artifacts[str(plan_path)]:
            raise ValueError("Wrong pair identity or plan checksum")
        if pair.digest(path.with_suffix(".npz")) != row["feature_sha256"]:
            raise ValueError("Pair feature checksum mismatch")
        current = dict(environment=row["inputs"]["environment"], lpips_sha256=row["inputs"]["lpips_sha256"],
                       detector_sha256=plan["detector_sha256"], code_hashes=plan["code_hashes"])
        if encoder is not None and current != encoder:
            raise ValueError("Mixed encoders/environments within pair")
        encoder = current
        if check_sources:
            for name, digest in row["inputs"]["sources"].items():
                if pair.digest(Path(name)) != digest:
                    raise ValueError(f"Changed video/GT: {name}")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
            if set(data.files) != set(features):
                raise ValueError("Wrong feature keys")
            for k in features:
                features[k].append(data[k])
        for k in values:
            values[k].append(row["scores"][k])
        artifacts[str(path)] = pair.digest(path)
        artifacts[str(path.with_suffix(".npz"))] = row["feature_sha256"]
    values = {k: np.array(v) for k, v in values.items()}
    features = {k: np.stack(v) for k, v in features.items()}
    validate_arrays(plan["items"], {KEEP: values["keepsake"], "baseline": values["baseline"]},
                    {KEEP: features["keepsake"], "baseline": features["baseline"], "GT": features["GT"]})
    if features["GT"].shape[1] != plan["config"]["clips_per_video"]:
        raise ValueError("Wrong clip count")
    return plan, values, features, encoder, artifacts


def extract_pair(args, duration, run):
    directory = args.output/"pairs"/f"{duration}s_{run}"
    expected = pair_args(args, duration, run, directory)
    old = args.root/"headline_uncertainty_180s_matched15"
    if duration == 180 and run == "baseline" and (old/"plan.json").exists():
        # Read-only import: never alter the successful prior job's artifacts.
        print(f"VERIFY prior matched extraction: {old}", flush=True)
        loaded = load_pair(old, expected, check_sources=True)
        return old, loaded
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(ROOT/"paper/paired_quality_uncertainty.py"), "extract"]
    for key in ("output", "duration", "expected_videos", "baseline", "keepsake", "manifest",
                "detector", "lpips_stride", "clips", "clip_stride"):
        command += ["--"+key.replace("_", "-"), str(getattr(expected, key))]
    print(f"EXTRACT {duration}s {run}: {directory / 'extract.log'}", flush=True)
    with (directory/"extract.log").open("a") as log:
        # Foreground child in the batch process group; TERM is forwarded below.
        with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as process:
            try:
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if code:
            raise RuntimeError(f"Extraction failed ({code}); see {directory / 'extract.log'}")
    return directory, load_pair(directory, expected, check_sources=True)


def export_group(args, name, pairs):
    duration, runs = GROUPS[name]
    values, features, sources = {}, {}, {}
    encoder = items = None
    for run in runs:
        if run == KEEP:
            continue
        plan, v, f, current, artifacts = pairs[(duration, run)]
        if (plan["duration_sec"] != duration or plan["config"] != config(duration)
                or plan["lpips_stride"] != (30 if duration == 60 else 90)):
            raise ValueError("Wrong duration or sampling protocol for report group")
        if encoder is not None and (current != encoder or items != plan["items"]):
            raise ValueError("Cross-policy encoder/runtime or cohort mismatch")
        encoder, items = current, plan["items"]
        if KEEP in values and (not np.array_equal(values[KEEP], v["keepsake"])
                or not np.array_equal(features[KEEP], f["keepsake"])
                or not np.array_equal(features["GT"], f["GT"])):
            raise ValueError("Repeated common reference features/scores differ")
        values.update({run: v["baseline"], KEEP: v["keepsake"]})
        features.update({run: f["baseline"], KEEP: f["keepsake"], "GT": f["GT"]})
        sources.update(artifacts)
    values = {r: values[r] for r in runs}
    mapping = read(args.cluster_map) if args.cluster_map else None
    out = args.output/name
    out.mkdir(exist_ok=True)
    protocol = dict(group=name, duration_sec=duration, sources=sources, draws=args.draws,
                    swaps=args.swaps, seed=args.seed, cluster_map=mapping, encoder=encoder,
                    suite_code_sha256=pair.digest(Path(__file__)))
    receipt = out/"receipt.json"
    if receipt.exists():
        record = read(receipt)
        if record["protocol"] != protocol:
            raise ValueError("Changed report inputs/settings; use a new suite output")
        for filename, digest in record["artifacts"].items():
            if pair.digest(out/filename) != digest:
                raise ValueError("Report artifact changed")
        print(f"REUSE REPORT {name}", flush=True)
        return
    # Verify low-rank FVD against the original covariance implementation once.
    checks = []
    evaluator = object.__new__(pair.FVDRunner)
    flat = {k: a.reshape(-1, a.shape[-1]) for k, a in features.items()}
    for run in runs:
        original = evaluator._frechet_distance(flat["GT"], flat[run])
        fast = pair.frechet_low_rank(flat["GT"], flat[run])
        if not np.isclose(original, fast, atol=.05, rtol=1e-6):
            raise ValueError("Numerical FVD discrepancy")
        checks.append(dict(run=run, covariance_fvd=original, low_rank_fvd=fast))
    print(f"BOOTSTRAP {name}: {args.draws} draws; {args.swaps} paired swaps/comparator", flush=True)
    summaries, contrasts, boot, labels = statistics(items, values, features, args.draws, args.swaps, args.seed, mapping)
    for row in summaries:
        legacy = PUBLISHED[name][row["run"]][0 if row["metric"] == "LPIPS" else 1]
        row.update(manuscript_estimate=legacy, recomputed_minus_manuscript=row["estimate"]-legacy)
    pair.write_csv(out/"summary.csv", summaries)
    pair.write_csv(out/"contrasts.csv", contrasts)
    pair.write_csv(out/"fvd_numerical_check.csv", checks)
    pair.write_csv(out/"per_video.csv", [dict(run=r, scene=i["scene"], start_frame=i["start_frame"],
        group=labels[j], lpips=float(values[r][j])) for r in runs for j, i in enumerate(items)])
    np.savez(out/"bootstrap.npz", scores=boot, runs=np.array(runs), metrics=np.array(["LPIPS", "FVD"]))
    rows = {(r["run"], r["metric"]): r for r in summaries}
    lines = [r"% Requires booktabs. Unadjusted paired trajectory-bootstrap percentile intervals.",
             r"\begin{tabular}{lrr}", r"\toprule", r"Method & LPIPS [95\% CI] & FVD [95\% CI] \\", r"\midrule"]
    for run in runs:
        cells = []
        for metric, digits in (("LPIPS", 4), ("FVD", 1)):
            r = rows[(run, metric)]
            cells.append(f"{r['estimate']:.{digits}f} [{r['ci_low']:.{digits}f}, {r['ci_high']:.{digits}f}]")
        lines.append(LABELS[run]+" & "+" & ".join(cells)+r" \\")
    (out/"table.tex").write_text("\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n")
    pair.save_json(out/"analysis.json", dict(protocol=protocol, resampling_groups=labels,
        ci="Unadjusted paired whole-group percentile bootstrap; pooled FVD recomputed per draw.",
        sign="difference = KEEPSAKE B32 minus comparator; negative favors B32. Reduction percent uses the opposite sign.",
        tests="Two-sided paired label swaps under within-group policy-label exchangeability; Holm across both metrics and all comparators in this table.",
        limitations="Post-hoc, existing outputs, not seed variance or held-out generalization. Finite-sample FVD bias remains. Default grouping treats trajectories as independent, not shared environments. CIs are not simultaneous; no equivalence claim."))
    pair.save_json(receipt, dict(protocol=protocol, artifacts={p.name: pair.digest(p) for p in out.iterdir() if p.is_file() and p != receipt}))
    print(f"COMPLETE {name}: {out/'table.tex'}", flush=True)


def freeze(args):
    paths = [Path(__file__), ROOT/"paper/paired_quality_uncertainty.py", ROOT/"paper/evidence_statistics.py",
             ROOT/"paper/finish_keepsake_pose_appearance.py", *sorted((ROOT/"utils").glob("*.py"))]
    plan = dict(groups=GROUPS, root=str(args.root.resolve()), detector=str(args.detector.resolve()),
                draws=args.draws, swaps=args.swaps, seed=args.seed,
                python=sys.version, numpy=np.__version__,
                cluster_map=read(args.cluster_map) if args.cluster_map else None,
                sources={str(p): pair.digest(p) for p in paths})
    for d in (60, 180):
        manifest = pair_args(args, d, "baseline", args.output).manifest
        plan["sources"][str(manifest)] = pair.digest(manifest)
        cohort(manifest, d)
    plan = json.loads(json.dumps(plan))
    path = args.output/"suite_plan.json"
    if path.exists() and read(path) != plan:
        raise ValueError("Suite code/protocol changed; choose a new output root")
    pair.save_json(path, plan)
    return plan


def assert_sources(sources):
    for path, digest in sources.items():
        if pair.digest(Path(path)) != digest:
            raise ValueError(f"Changed source: {path}")


def run(args):
    plan = freeze(args)
    pairs, statuses = {}, []
    for duration, runs in GROUPS.values():
        for policy in runs:
            key = duration, policy
            if policy == KEEP or key in pairs:
                continue
            assert_sources(plan["sources"])
            try:
                if args.phase == "report":
                    pointer = read(args.output/"pair_index"/f"{duration}s_{policy}.json")
                    loaded = load_pair(Path(pointer["path"]))
                    if loaded[-1] != pointer["artifacts"]:
                        raise ValueError("Pair inputs changed since extraction")
                else:
                    directory, loaded = extract_pair(args, duration, policy)
                    pair.save_json(args.output/"pair_index"/f"{duration}s_{policy}.json",
                                   dict(path=str(directory.resolve()), artifacts=loaded[-1]))
                pairs[key] = loaded
            except Exception as exc:
                print(f"FAILED PAIR {key}: {exc}", flush=True)
                statuses.append(dict(stage=f"pair_{duration}s_{policy}", status="failed", error=str(exc)))
            pair.save_json(args.output/"status.json", dict(status="running", pairs_loaded=len(pairs), stages=statuses))
    for name, (duration, runs) in GROUPS.items():
        try:
            assert_sources(plan["sources"])
            export_group(args, name, pairs)
            statuses.append(dict(stage=name, status="complete", error=""))
        except Exception as exc:
            print(f"FAILED REPORT {name}: {exc}", flush=True)
            statuses.append(dict(stage=name, status="failed", error=str(exc)))
    # Existing scalar files are optional supplementary evidence, never substituted
    # for the freshly matched LPIPS/FVD extraction above.
    try:
        from paper.evidence_statistics import export_scalars
        folder = args.output/"legacy_scalar_60s"
        folder.mkdir(exist_ok=True)
        metadata = export_scalars(args.root/"budget_metrics_60s_820776/scores.csv",
                                  cohort(pair_args(args, 60, "baseline", args.output).manifest, 60), folder)
        pair.save_json(folder/"provenance.json", metadata)
        statuses.append(dict(stage="legacy_scalar_60s", status="complete", error=""))
    except Exception as exc:
        statuses.append(dict(stage="legacy_scalar_60s", status="unavailable", error=str(exc)))
    complete = all(any(r["stage"] == name and r["status"] == "complete" for r in statuses) for name in GROUPS)
    if complete:
        import csv
        rows = []
        for name in GROUPS:
            with (args.output/name/"contrasts.csv").open() as handle:
                rows.extend(dict(family=name, **r) for r in csv.DictReader(handle))
        if len(rows) != 18:
            raise ValueError("Expected all 18 MemCam quality contrasts")
        for row, p in zip(rows, holm([float(r["p_two_sided"]) for r in rows])):
            row["p_holm_all_18"] = p
        pair.write_csv(args.output/"all_quality_contrasts.csv", rows)
    pair.write_csv(args.output/"coverage.csv", statuses)
    assert_sources(plan["sources"])
    pair.save_json(args.output/"status.json", dict(status="complete" if complete else "incomplete",
        quality_groups_complete=complete, stages=statuses,
        not_covered=["WorldMem (separate handoff)", "Native round-trip protocol comparability and published baseline uncertainty",
                     "180s VBench budget provenance", "n=1 timing inference", "New-seed variation"]))
    with zipfile.ZipFile(args.output/"reports.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.output.rglob("*")):
            if path.is_file() and path.suffix in (".csv", ".tex", ".json", ".npz"):
                archive.write(path, path.relative_to(args.output))
    print(f"Quality groups: {'complete' if complete else 'incomplete'}; reports: {args.output/'reports.zip'}", flush=True)
    if not complete:
        raise RuntimeError(f"Some quality groups failed; see {args.output/'coverage.csv'}. Completed groups and pairs are preserved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "report"))
    parser.add_argument("--root", type=Path, default=Path.home()/"memcam_results")
    parser.add_argument("--output", type=Path, default=Path.home()/"memcam_results/headline_intervals_all")
    parser.add_argument("--detector", type=Path, default=Path.home()/"hf_cache/memcam_fvd/i3d_torchscript.pt")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--swaps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cluster-map", type=Path)
    args = parser.parse_args()
    if args.draws < 2 or args.swaps < 1:
        parser.error("Invalid resampling counts")
    if args.phase == "run" and not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run metric extraction in a Slurm GPU allocation, not on a login node")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"pair_index").mkdir(exist_ok=True)
    def interrupted(signum, frame):
        raise InterruptedError(f"Interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (args.output/".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args)
        except BaseException as exc:
            prior = read(args.output/"status.json") if (args.output/"status.json").exists() else {}
            pair.save_json(args.output/"status.json", dict(prior, status="failed", error=str(exc)))
            traceback.print_exc()
            raise SystemExit(1)


if __name__ == "__main__":
    main()
