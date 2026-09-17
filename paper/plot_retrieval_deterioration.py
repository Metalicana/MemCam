"""Plot measured retrieval deterioration from an existing decomposition CSV (CPU).

Queries are averaged within sections, then trajectories receive equal weight.
No manuscript estimates or precomputed figure coordinates are embedded here.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


FIELDS = ("selected_view_mismatch", "selected_memory_corruption",
          "selected_effective_mismatch", "full_oracle_effective_mismatch")
NAMES = ("View mismatch", "Stored-content corruption", "Selected effective mismatch", "Best available mismatch")
IDENTITY = ("row", "scene", "dataset_start_frame", "duration_sec")


def load_sections(path, run, duration, expected_videos):
    grouped, seen, times = defaultdict(list), set(), defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {*IDENTITY, *FIELDS, "run_name", "section_idx", "target_frame"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            if row["run_name"] != run or int(row["duration_sec"]) != duration:
                continue
            identity = tuple(row[k] for k in IDENTITY)
            section, target = int(row["section_idx"]), int(row["target_frame"])
            key = (*identity, section, target)
            if key in seen:
                raise ValueError(f"Duplicate query: {key}")
            seen.add(key)
            values = np.asarray([float(row[k]) for k in FIELDS])
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite metric in query {key}")
            if values[3] > values[2] + 1e-5:
                raise ValueError(f"Hindsight best exceeds selected mismatch in query {key}")
            if row.get("candidate_count_mismatch") and float(row["candidate_count_mismatch"]) != 0:
                raise ValueError(f"Candidate-bank reconstruction mismatch in query {key}")
            grouped[(identity, section)].append(values)
            times[(identity, section)].append(target)
    identities = sorted({key[0] for key in grouped})
    if len(identities) != expected_videos:
        raise ValueError(f"Expected {expected_videos} trajectories for {run}/{duration}s; found {len(identities)}")
    sections = sorted(s for identity, s in grouped if identity == identities[0])
    if len(sections) < 8 or any(b != a + 1 for a, b in zip(sections, sections[1:])):
        raise ValueError("Need at least eight contiguous sections per trajectory")
    for identity in identities:
        if sorted(s for i, s in grouped if i == identity) != sections:
            raise ValueError(f"Section coverage differs for {identity}; refusing a changing cohort")
    values = np.asarray([[np.mean(grouped[(i, s)], axis=0) for s in sections] for i in identities])
    targets = np.asarray([[np.mean(times[(i, s)]) for s in sections] for i in identities])
    return identities, sections, values, targets, len(seen)


def summarize(values, targets, fps, bins, repeats, seed):
    n, sections, _ = values.shape
    if n < 2 or not 2 <= bins <= sections or repeats < 100 or fps <= 0:
        raise ValueError("Need >=2 trajectories, 2..Nsection bins, >=100 bootstrap draws, positive FPS")
    quarter = sections // 4
    deltas = values[:, -quarter:].mean(axis=1) - values[:, :quarter].mean(axis=1)
    groups = np.array_split(np.arange(sections), bins)
    curves = np.stack([values[:, indices].mean(axis=1) for indices in groups], axis=1)
    x = np.asarray([targets[:, indices].mean() / fps for indices in groups])
    draws = np.random.default_rng(seed).integers(0, n, size=(repeats, n))
    def interval(array):
        samples = array[draws].mean(axis=1)
        return np.quantile(samples, [.025, .975], axis=0)
    return {"quarter_sections": quarter, "deltas": deltas, "delta_mean": deltas.mean(axis=0),
            "delta_ci": interval(deltas), "curve_mean": curves.mean(axis=0),
            "curve_ci": interval(curves), "time_sec": x}


def plot(result, n, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    for index, color in enumerate(("#238575", "#BD443A")):
        y = 1 - index
        mean = result["delta_mean"][index]
        lo, hi = result["delta_ci"][:, index]
        ax.scatter(result["deltas"][:, index], y + np.linspace(-.1, .1, n),
                   s=16, color=color, alpha=.3)
        ax.plot([lo, hi], [y, y], color=color, linewidth=2.5)
        ax.scatter([mean], [y], color=color, s=48, zorder=3)
        ax.annotate(f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]", (mean, y),
                    xytext=(0, 18), textcoords="offset points", ha="center", fontsize=9)
    ax.axvline(0, color="#777777", linewidth=1, linestyle="--")
    ax.set(yticks=[1, 0], yticklabels=["View\nmismatch", "Memory\ncorruption"], ylim=(-.45, 1.55),
           xlabel="Late minus early DINO distance")
    ax.margins(x=.3)
    ax.set_title("(a) Changes in retrieved memory", loc="left", fontsize=11)
    ax = axes[1]
    for index, name, color in ((2, "Selected memory", "#BD443A"), (3, "Best available in archive", "#2673A8")):
        ax.plot(result["time_sec"], result["curve_mean"][:, index], marker="o", markersize=4,
                color=color, label=name, linewidth=2)
        ax.fill_between(result["time_sec"], result["curve_ci"][0, :, index],
                        result["curve_ci"][1, :, index], color=color, alpha=.12)
    ax.set(xlabel="Rollout time (s)", ylabel="DINO distance to target GT")
    ax.set_title("(b) Selected versus available evidence", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"retrieval_deterioration.{ext}", dpi=200)
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path.home() / "memcam_results/context_180s/unbounded_failure_decomposition_180s/tables/query_decomposition.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", default="baseline")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--expected-videos", type=int, default=15)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Missing source CSV: {args.input}. Supply --input with the existing query_decomposition.csv; no GPU analysis is launched.")
    if args.expected_videos < 2:
        parser.error("At least two trajectories are required for uncertainty estimates")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    identities, sections, values, targets, queries = load_sections(args.input, args.run, args.duration, args.expected_videos)
    result = summarize(values, targets, args.fps, args.bins, args.bootstrap, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    plot(result, len(identities), args.output)
    summary = [{"metric": field, "late_minus_early": float(result["delta_mean"][i]),
                "ci_low": float(result["delta_ci"][0, i]), "ci_high": float(result["delta_ci"][1, i])}
               for i, field in enumerate(FIELDS)]
    write_csv(args.output / "changes.csv", summary)
    write_csv(args.output / "trajectory_changes.csv", [dict(zip(IDENTITY, identity), **dict(zip(FIELDS, delta)))
                                                       for identity, delta in zip(identities, result["deltas"])])
    write_csv(args.output / "curves.csv", [
        {"time_sec": float(t), "metric": field, "mean": float(result["curve_mean"][b, i]),
         "ci_low": float(result["curve_ci"][0, b, i]), "ci_high": float(result["curve_ci"][1, b, i])}
        for b, t in enumerate(result["time_sec"]) for i, field in enumerate(FIELDS)])
    provenance = {"source": str(args.input.resolve()), "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "trajectories": [dict(zip(IDENTITY, identity)) for identity in identities],
                  "queries": queries, "sections_total": len(identities) * len(sections),
                  "section_indices": sections, "early_sections": sections[:result["quarter_sections"]],
                  "late_sections": sections[-result["quarter_sections"]:],
                  "method": "Query means within sections; equal section weights within each trajectory. First/last floor(Nsections/4) sections define early/late. Curves use equal-count ordered-section bins. Paired trajectory bootstrap; pointwise percentile 95% intervals. FPS is an explicit input.",
                  "interpretation": "Observational temporal association, not causal error propagation or a candidate-count intervention. A CI crossing zero does not establish equivalence or prove flatness."}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output / "caption.txt").write_text(
        f"Unbounded-memory retrieval diagnostics on {len(identities)} trajectories ({queries:,} queries). "
        "(a) Late-minus-early changes in selected view mismatch and stored-content corruption; faint dots show individual trajectories. "
        "(b) Effective mismatch of selected memory and the hindsight-best available archive item over rollout time. "
        "All distances use DINO features; lower is better. Intervals resample trajectories (95%); bands are pointwise. "
        "These diagnostics characterize retrieval, not causal error propagation. See provenance.json for aggregation and cohort details.\n")
    print(f"Source: {args.input}\n{len(identities)} trajectories; {queries:,} queries; {len(identities) * len(sections):,} sections")
    for name, row in zip(NAMES, summary):
        print(f"{name:28s} {row['late_minus_early']:+.4f}  95% CI [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]")
    print(f"Figure: {args.output / 'retrieval_deterioration.pdf'}")


if __name__ == "__main__":
    main()
