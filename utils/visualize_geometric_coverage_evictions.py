"""Visualize completed Geometric Coverage eviction decisions.

The generation trace supplies the actual evicted frame identities. For one
informative section, this utility reconstructs the pre-eviction bank and
recomputes the exact pose/DINO affinity used by ``slam_covisibility``. It does
not rerun video generation or change the retrieval rule.
"""

import argparse
import csv
import importlib.util
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = REPO_ROOT / "utils" / "visualize_coverage_hysteresis.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COLORS = {
    "retained_old": "#2878b5",
    "retained_new": "#2b9b58",
    "evicted_old": "#a33d3d",
    "evicted_new": "#e04b3f",
    "anchor": "#7656a6",
}


def load_common_module():
    spec = importlib.util.spec_from_file_location(
        "memcam_geometric_visualization_common",
        COMMON_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_int(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_float(value):
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def geometric_evictions_by_section(events):
    grouped = defaultdict(list)
    for row in events:
        if row.get("event") != "memory_eviction":
            continue
        if row.get("memory_policy") != "slam_covisibility":
            continue
        section_idx = safe_int(row.get("section_idx"))
        frame_idx = safe_int(row.get("evicted_memory_frame"))
        if section_idx is None or frame_idx is None:
            continue
        grouped[section_idx].append(row)
    return grouped


def reconstruct_geometric_snapshots(events, budget=32, frames_per_section=77):
    """Reconstruct candidate and retained banks from actual trace evictions."""
    evictions = geometric_evictions_by_section(events)
    if not evictions:
        return {}

    retained = [0]
    snapshots = {}
    for section_idx in range(max(evictions) + 1):
        rows = evictions.get(section_idx, [])
        section_start = section_idx * (int(frames_per_section) - 1)
        section_end = section_start + int(frames_per_section) - 1
        incoming = [
            frame_idx
            for frame_idx in range(section_start, section_end + 1)
            if frame_idx not in retained
        ]
        prospective = list(dict.fromkeys(retained + incoming))
        evicted = [int(row["evicted_memory_frame"]) for row in rows]
        evicted_set = set(evicted)
        next_retained = [idx for idx in prospective if idx not in evicted_set]
        snapshots[section_idx] = {
            "section_idx": section_idx,
            "section_start": section_start,
            "section_end": section_end,
            "current_memory": list(retained),
            "incoming": incoming,
            "prospective": prospective,
            "eviction_rows": rows,
            "evicted": evicted,
            "retained": next_retained,
        }
        retained = next_retained

    malformed = [
        idx
        for idx, snapshot in snapshots.items()
        if len(snapshot["retained"]) > int(budget)
    ]
    if malformed:
        raise RuntimeError(
            "Could not reconstruct the Geometric Coverage bank within budget "
            f"for sections {malformed[:5]}"
        )
    return snapshots


def trace_information_score(events):
    rows = [
        row
        for values in geometric_evictions_by_section(events).values()
        for row in values
    ]
    observers = [safe_float(row.get("eviction_covisible_observers")) for row in rows]
    observers = [value for value in observers if value is not None]
    return len(rows) + 2.0 * sum(value >= 3 for value in observers)


def choose_trace(common, manifest_rows, root, run_name, requested_row=None):
    candidates = []
    for row_idx, item in manifest_rows.items():
        if requested_row is not None and int(row_idx) != int(requested_row):
            continue
        trace_path = common.find_trace(root, run_name, item)
        video_path = Path(root) / run_name / f"{item['output_prefix']}custom.mp4"
        if trace_path is None or not video_path.is_file():
            continue
        events = common.load_trace(trace_path)
        if not geometric_evictions_by_section(events):
            continue
        candidates.append(
            (trace_information_score(events), row_idx, item, trace_path, events)
        )
    if not candidates:
        suffix = "" if requested_row is None else f" for row {requested_row}"
        raise RuntimeError(
            "No completed slam_covisibility trace and video found" + suffix
        )
    return max(candidates, key=lambda row: (row[0], row[1]))


def choose_section(snapshots, requested_section=None):
    if requested_section is not None:
        section_idx = int(requested_section)
        if section_idx not in snapshots:
            raise ValueError(f"Section {section_idx} is absent from the trace")
        return section_idx

    max_section = max(snapshots)
    candidates = []
    for section_idx, snapshot in snapshots.items():
        if section_idx == 0 or not snapshot["eviction_rows"]:
            continue
        observers = [
            safe_float(row.get("eviction_covisible_observers"))
            for row in snapshot["eviction_rows"]
        ]
        observers = [value for value in observers if value is not None]
        old_evictions = sum(
            idx in set(snapshot["current_memory"]) for idx in snapshot["evicted"]
        )
        score = float(np.mean(observers)) if observers else 0.0
        score += 0.2 * old_evictions
        score += 0.5 * section_idx / max(max_section, 1)
        candidates.append((score, section_idx))
    if not candidates:
        raise RuntimeError("Trace has no mature Geometric Coverage section")
    return max(candidates)[1]


def frame_status(frame_idx, snapshot):
    frame_idx = int(frame_idx)
    if frame_idx == 0:
        return "anchor"
    is_new = frame_idx in set(snapshot["incoming"])
    is_evicted = frame_idx in set(snapshot["evicted"])
    if is_evicted:
        return "evicted_new" if is_new else "evicted_old"
    return "retained_new" if is_new else "retained_old"


def classical_mds(dissimilarity, dimensions=2):
    dissimilarity = np.asarray(dissimilarity, dtype=np.float64)
    if dissimilarity.ndim != 2 or dissimilarity.shape[0] != dissimilarity.shape[1]:
        raise ValueError("dissimilarity must be a square matrix")
    count = dissimilarity.shape[0]
    if count == 0:
        return np.zeros((0, dimensions), dtype=np.float64)
    centering = np.eye(count) - np.ones((count, count), dtype=np.float64) / count
    gram = -0.5 * centering @ (dissimilarity ** 2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    positive = [idx for idx in order if eigenvalues[idx] > 1e-12][:dimensions]
    coordinates = np.zeros((count, dimensions), dtype=np.float64)
    for column, idx in enumerate(positive):
        coordinates[:, column] = eigenvectors[:, idx] * np.sqrt(eigenvalues[idx])
    return coordinates


def recent_anchor_bank(frames, budget, anchor=0):
    frames = list(dict.fromkeys(int(idx) for idx in frames))
    if int(budget) < 1:
        raise ValueError("budget must be positive")
    non_anchor = [idx for idx in frames if idx != int(anchor)]
    if int(anchor) in frames:
        return [int(anchor)] + non_anchor[-max(int(budget) - 1, 0) :]
    return non_anchor[-int(budget) :]


def bank_support(similarity, all_frames, bank_frames):
    similarity = np.asarray(similarity, dtype=np.float64)
    positions = {int(frame_idx): idx for idx, frame_idx in enumerate(all_frames)}
    columns = [positions[int(frame_idx)] for frame_idx in bank_frames]
    if not columns:
        return np.zeros(len(all_frames), dtype=np.float64)
    return np.max(similarity[:, columns], axis=1)


def support_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "p10": 0.0, "minimum": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p10": float(np.percentile(values, 10)),
        "minimum": float(np.min(values)),
    }


def nearest_retained_pairs(snapshot, affinity, details, limit=5):
    frames = snapshot["prospective"]
    positions = {frame_idx: idx for idx, frame_idx in enumerate(frames)}
    retained = snapshot["retained"]
    candidates = []
    for evicted in snapshot["evicted"]:
        if not retained:
            continue
        row = affinity[positions[evicted], [positions[idx] for idx in retained]]
        nearest_position = int(np.argmax(row))
        nearest = int(retained[nearest_position])
        value = float(row[nearest_position])
        candidates.append(
            (
                -value,
                float(details[evicted]["score"]),
                evicted,
                nearest,
                value,
            )
        )
    candidates.sort()
    selected = []
    used_nearest = set()
    for _, _, evicted, nearest, value in candidates:
        if nearest in used_nearest and len(candidates) > int(limit):
            continue
        selected.append((evicted, nearest, value))
        used_nearest.add(nearest)
        if len(selected) >= int(limit):
            break
    return selected


def render_camera_space(common, c2ws, snapshot, pairs, output_path):
    plt = common.configure_matplotlib()
    from matplotlib.lines import Line2D

    positions = np.asarray(c2ws[:, :3, 3], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        color="#d2d2d2",
        linewidth=1.0,
        alpha=0.7,
        label="camera trajectory",
    )
    for evicted, retained, _ in pairs:
        ax.plot(
            [positions[evicted, 0], positions[retained, 0]],
            [positions[evicted, 1], positions[retained, 1]],
            color="#999999",
            linewidth=1.0,
            linestyle="--",
            alpha=0.7,
        )
    for status in COLORS:
        frames = [
            idx for idx in snapshot["prospective"] if frame_status(idx, snapshot) == status
        ]
        if not frames:
            continue
        marker = "*" if status == "anchor" else ("x" if "evicted" in status else "o")
        ax.scatter(
            positions[frames, 0],
            positions[frames, 1],
            color=COLORS[status],
            marker=marker,
            s=125 if marker in {"*", "x"} else 58,
            linewidths=2.0 if marker == "x" else 0.8,
            edgecolors="none" if marker == "x" else "white",
            alpha=0.9,
            zorder=3,
        )
    retained = snapshot["retained"]
    forward = np.asarray(c2ws[retained, :3, 0], dtype=np.float64)
    planar_norm = np.linalg.norm(forward[:, :2], axis=1, keepdims=True)
    forward_xy = forward[:, :2] / np.maximum(planar_norm, 1e-12)
    span = max(np.ptp(positions[:, 0]), np.ptp(positions[:, 1]), 1.0)
    ax.quiver(
        positions[retained, 0],
        positions[retained, 1],
        forward_xy[:, 0],
        forward_xy[:, 1],
        color="#16324f",
        angles="xy",
        scale_units="xy",
        scale=10.0 / span,
        width=0.003,
        alpha=0.48,
        zorder=2,
    )
    handles = [
        Line2D([0], [0], color="#d2d2d2", label="full camera path"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["retained_old"], label="retained old", markersize=8),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["retained_new"], label="retained new", markersize=8),
        Line2D([0], [0], marker="x", color=COLORS["evicted_new"], label="evicted", markersize=8, linestyle="None"),
        Line2D([0], [0], color="#999999", linestyle="--", label="evicted -> retained substitute"),
    ]
    ax.legend(handles=handles, frameon=False, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("camera x")
    ax.set_ylabel("camera y")
    ax.set_title(
        f"Physical camera space before and after eviction (section {snapshot['section_idx']})",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(color="#ededed", linewidth=0.7)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_affinity_space(common, snapshot, affinity, pairs, output_path):
    plt = common.configure_matplotlib()
    frames = snapshot["prospective"]
    coordinates = classical_mds(1.0 - affinity)
    positions = {frame_idx: idx for idx, frame_idx in enumerate(frames)}
    fig, ax = plt.subplots(figsize=(11, 8))
    for evicted, retained, _ in pairs:
        a = coordinates[positions[evicted]]
        b = coordinates[positions[retained]]
        ax.plot([a[0], b[0]], [a[1], b[1]], "--", color="#a3a3a3", linewidth=1.0)
    for status in COLORS:
        selected = [idx for idx in frames if frame_status(idx, snapshot) == status]
        if not selected:
            continue
        xy = coordinates[[positions[idx] for idx in selected]]
        marker = "*" if status == "anchor" else ("x" if "evicted" in status else "o")
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            color=COLORS[status],
            marker=marker,
            s=130 if marker in {"*", "x"} else 58,
            linewidths=2.0 if marker == "x" else 0.8,
            edgecolors="none" if marker == "x" else "white",
            alpha=0.9,
            label=status.replace("_", " "),
        )
    for evicted, retained, value in pairs:
        point = coordinates[positions[evicted]]
        ax.annotate(
            f"f{evicted} -> f{retained}\nK={value:.2f}",
            point,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.legend(frameon=False, loc="best")
    ax.set_xlabel("joint-affinity MDS 1")
    ax.set_ylabel("joint-affinity MDS 2")
    ax.set_title(
        "Policy space: nearby points share camera pose and visual content",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.99,
        "MDS of 1 - K, where K = 0.65 pose affinity + 0.35 DINO cosine",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        color="#444444",
    )
    ax.grid(color="#ededed", linewidth=0.7)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_score_decomposition(common, snapshot, details, output_path):
    plt = common.configure_matplotlib()
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    for status in COLORS:
        frames = [
            idx for idx in snapshot["prospective"] if frame_status(idx, snapshot) == status
        ]
        if not frames:
            continue
        marker = "*" if status == "anchor" else ("x" if "evicted" in status else "o")
        ax.scatter(
            [details[idx]["covisible_observers"] for idx in frames],
            [details[idx]["unique_bonus"] for idx in frames],
            s=[55 + 80 * details[idx]["max_covisibility"] for idx in frames],
            color=COLORS[status],
            marker=marker,
            linewidths=2.0 if marker == "x" else 0.8,
            edgecolors="none" if marker == "x" else "white",
            alpha=0.86,
            label=status.replace("_", " "),
        )
    low_frames = sorted(
        snapshot["evicted"],
        key=lambda idx: details[idx]["score"],
    )[:6]
    for frame_idx in low_frames:
        ax.annotate(
            f"f{frame_idx}\nU={details[frame_idx]['score']:.2f}",
            (
                details[frame_idx]["covisible_observers"],
                details[frame_idx]["unique_bonus"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(3, color="#555555", linestyle="--", linewidth=1.2)
    ax.text(3.05, ax.get_ylim()[1] * 0.95, "3 observers saturates redundancy penalty", fontsize=9, va="top")
    ax.set_xlabel("number of co-visible memories (K >= 0.65)")
    ax.set_ylabel("unique bonus = 1 - nearest affinity")
    ax.set_title(
        "Why a frame is evicted: many observers and a close substitute",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.99,
        "U = (1 - redundancy) + 0.5/(observers + 1) + 0.25 x unique bonus; lowest U is evicted",
        transform=ax.transAxes,
        va="top",
        fontsize=9.5,
        color="#444444",
    )
    ax.legend(frameon=False, loc="best")
    ax.grid(color="#ededed", linewidth=0.7)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_pair_montage(common, pairs, pair_rows, images, output_path):
    if not pairs:
        return None
    plt = common.configure_matplotlib()
    fig, axes = plt.subplots(len(pairs), 3, figsize=(13.2, 3.5 * len(pairs)), squeeze=False)
    for row_idx, (evicted, retained, _) in enumerate(pairs):
        row = pair_rows[(evicted, retained)]
        axes[row_idx, 0].imshow(common.image_array(images, evicted))
        axes[row_idx, 1].imshow(common.image_array(images, retained))
        axes[row_idx, 0].set_title(f"evicted f{evicted}", color=COLORS["evicted_new"], fontweight="bold")
        axes[row_idx, 1].set_title(f"retained substitute f{retained}", color=COLORS["retained_old"], fontweight="bold")
        axes[row_idx, 2].axis("off")
        axes[row_idx, 2].text(
            0.02,
            0.92,
            "\n".join(
                [
                    f"pose affinity     {row['pose_affinity']:.3f}",
                    f"DINO cosine      {row['dino_cosine']:.3f}",
                    f"joint K          {row['joint_affinity']:.3f}",
                    f"co-visible count {row['covisible_observers']}",
                    f"retention U      {row['retention_score']:.3f}",
                ]
            ),
            va="top",
            family="monospace",
            fontsize=11,
        )
        for ax in axes[row_idx, :2]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle(
        "Actual low-score evictions and the memories that already cover them",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_coverage_comparison(common, coverage, output_path):
    plt = common.configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    colors = {"Geometric Coverage-32": "#2878b5", "Recent-32 + anchor": "#d65f4a"}
    labels = {
        "camera": "direct camera-FOV support",
        "joint": "policy pose+DINO support",
    }
    for ax, metric in zip(axes, ("camera", "joint")):
        for bank_name, values in coverage[metric].items():
            x = np.linspace(0.0, 1.0, 201)
            survival = np.asarray([np.mean(values >= threshold) for threshold in x])
            summary = support_summary(values)
            ax.plot(
                x,
                survival,
                linewidth=2.4,
                color=colors[bank_name],
                label=f"{bank_name} (mean {summary['mean']:.3f}, p10 {summary['p10']:.3f})",
            )
        ax.set_xlabel("required support")
        ax.set_ylabel("fraction of candidate views covered")
        ax.set_title(labels[metric], fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(color="#ededed", linewidth=0.7)
        ax.legend(frameon=False, fontsize=9)
        for name in ("top", "right"):
            ax.spines[name].set_visible(False)
    fig.suptitle(
        "Does the retained 32-frame bank preserve coverage?",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_timeline(common, snapshots, output_path):
    plt = common.configure_matplotlib()
    fig, ax = plt.subplots(figsize=(14, 7))
    for section_idx, snapshot in sorted(snapshots.items()):
        ax.scatter(
            [section_idx] * len(snapshot["retained"]),
            snapshot["retained"],
            s=9,
            color=COLORS["retained_old"],
            alpha=0.42,
            linewidths=0,
        )
        old = [idx for idx in snapshot["evicted"] if idx in snapshot["current_memory"]]
        new = [idx for idx in snapshot["evicted"] if idx in snapshot["incoming"]]
        ax.scatter([section_idx] * len(old), old, marker="x", s=22, color=COLORS["evicted_old"], alpha=0.75)
        ax.scatter([section_idx] * len(new), new, marker="x", s=22, color=COLORS["evicted_new"], alpha=0.75)
    ax.set_xlabel("generation section")
    ax.set_ylabel("memory frame index")
    ax.set_title("Geometric Coverage keeps sparse anchors across the full trajectory", fontsize=17, fontweight="bold")
    ax.grid(color="#ededed", linewidth=0.75)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize completed Geometric Coverage evictions."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run_name", default="slam_b32_covisibility")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--frames_per_section", type=int, default=77)
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--section", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_examples", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    common = load_common_module()
    manifest_rows = common.load_manifest(args.manifest, args.duration)
    _, row_idx, item, trace_path, events = choose_trace(
        common,
        manifest_rows,
        args.root,
        args.run_name,
        requested_row=args.row,
    )
    snapshots = reconstruct_geometric_snapshots(
        events,
        budget=args.budget,
        frames_per_section=args.frames_per_section,
    )
    section_idx = choose_section(snapshots, args.section)
    snapshot = snapshots[section_idx]

    policy_module = common.load_policy_module()
    from dataset.poses import load_c2ws_from_json

    c2ws = load_c2ws_from_json(
        item["pose_path"],
        start_frame=int(item["start_frame"]),
        num_frames=int(item["num_frames"]),
    )
    video_path = common.resolve_video_path(args.root, args.run_name, item)
    images = common.load_video_frames(video_path, snapshot["prospective"])
    missing = [idx for idx in snapshot["prospective"] if idx not in images]
    if missing:
        raise RuntimeError(f"Could not decode frames: {missing[:10]}")

    print(
        f"Encoding {len(snapshot['prospective'])} candidate frames with DINO "
        f"on {args.device}"
    )
    extractor = policy_module.VisualMemoryFeatureExtractor(device=args.device)
    dino_batch, rgb_batch = extractor.encode_pil_images(
        [images[idx] for idx in snapshot["prospective"]]
    )
    dino_features = {
        idx: dino_batch[position]
        for position, idx in enumerate(snapshot["prospective"])
    }
    rgb_features = {
        idx: rgb_batch[position]
        for position, idx in enumerate(snapshot["prospective"])
    }
    _, details = policy_module.compute_slam_covisibility_scores(
        memory_frame_indices=snapshot["prospective"],
        c2ws=c2ws,
        pinned_frames={0},
        dino_features=dino_features,
        rgb_features=rgb_features,
        return_details=True,
    )
    affinity = policy_module._slam_covisibility_affinity(
        memory_frame_indices=snapshot["prospective"],
        c2ws=c2ws,
        dino_features=dino_features,
        rgb_features=rgb_features,
        self_similarity=0.0,
    )

    prospective_order = {idx: position for position, idx in enumerate(snapshot["prospective"])}
    protected = {0, snapshot["section_end"]}
    recomputed_evicted = {
        idx
        for idx in sorted(
            (idx for idx in snapshot["prospective"] if idx not in protected),
            key=lambda idx: (details[idx]["score"], prospective_order[idx]),
        )[: max(0, len(snapshot["prospective"]) - args.budget)]
    }
    traced_evicted = set(snapshot["evicted"])
    overlap = (
        len(traced_evicted & recomputed_evicted) / len(traced_evicted)
        if traced_evicted
        else 1.0
    )

    pairs = nearest_retained_pairs(
        snapshot,
        affinity,
        details,
        limit=args.max_examples,
    )
    positions = {idx: position for position, idx in enumerate(snapshot["prospective"])}
    pose_distance = policy_module.pose_distances(
        c2ws,
        snapshot["prospective"],
        snapshot["prospective"],
    )
    dino_matrix = np.stack([dino_features[idx] for idx in snapshot["prospective"]])
    dino_matrix = dino_matrix / np.maximum(np.linalg.norm(dino_matrix, axis=1, keepdims=True), 1e-12)
    dino_cosine = np.clip(dino_matrix @ dino_matrix.T, -1.0, 1.0)
    pair_rows = {}
    for evicted, retained, value in pairs:
        i, j = positions[evicted], positions[retained]
        pair_rows[(evicted, retained)] = {
            "row": row_idx,
            "scene": item["scene"],
            "section_idx": section_idx,
            "evicted_frame": evicted,
            "retained_substitute": retained,
            "pose_affinity": float(np.exp(-pose_distance[i, j])),
            "dino_cosine": float(dino_cosine[i, j]),
            "joint_affinity": value,
            "covisible_observers": int(details[evicted]["covisible_observers"]),
            "retention_score": float(details[evicted]["score"]),
        }

    camera_similarity = policy_module.camera_trajectory_similarity(
        c2ws,
        snapshot["prospective"],
        snapshot["prospective"],
    )
    np.fill_diagonal(camera_similarity, 0.0)
    recent = recent_anchor_bank(snapshot["prospective"], args.budget, anchor=0)
    banks = {
        "Geometric Coverage-32": snapshot["retained"],
        "Recent-32 + anchor": recent,
    }
    coverage = {"camera": {}, "joint": {}}
    coverage_rows = []
    for bank_name, bank in banks.items():
        coverage["camera"][bank_name] = bank_support(
            camera_similarity,
            snapshot["prospective"],
            bank,
        )
        coverage["joint"][bank_name] = bank_support(
            affinity,
            snapshot["prospective"],
            bank,
        )
        for metric_name in ("camera", "joint"):
            summary = support_summary(coverage[metric_name][bank_name])
            coverage_rows.append(
                {
                    "row": row_idx,
                    "scene": item["scene"],
                    "section_idx": section_idx,
                    "bank": bank_name,
                    "metric": metric_name,
                    "bank_size": len(bank),
                    **summary,
                }
            )

    figures = args.output_dir / "figures"
    tables = args.output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    outputs = [
        render_camera_space(
            common,
            c2ws,
            snapshot,
            pairs,
            figures / f"01_camera_space_section_{section_idx}.png",
        ),
        render_affinity_space(
            common,
            snapshot,
            affinity,
            pairs,
            figures / f"02_joint_affinity_space_section_{section_idx}.png",
        ),
        render_score_decomposition(
            common,
            snapshot,
            details,
            figures / f"03_score_decomposition_section_{section_idx}.png",
        ),
        render_pair_montage(
            common,
            pairs,
            pair_rows,
            images,
            figures / f"04_evicted_substitutes_section_{section_idx}.png",
        ),
        render_coverage_comparison(
            common,
            coverage,
            figures / f"05_coverage_comparison_section_{section_idx}.png",
        ),
        render_timeline(
            common,
            snapshots,
            figures / "06_geometric_memory_timeline.png",
        ),
    ]

    score_rows = []
    for frame_idx in snapshot["prospective"]:
        score_rows.append(
            {
                "row": row_idx,
                "scene": item["scene"],
                "section_idx": section_idx,
                "memory_frame": frame_idx,
                "status": frame_status(frame_idx, snapshot),
                "trace_evicted": int(frame_idx in traced_evicted),
                "recomputed_evicted": int(frame_idx in recomputed_evicted),
                **details[frame_idx],
            }
        )
    write_csv(tables / "geometric_section_scores.csv", score_rows)
    write_csv(tables / "evicted_substitute_pairs.csv", list(pair_rows.values()))
    write_csv(tables / "coverage_comparison.csv", coverage_rows)

    geo_camera = support_summary(coverage["camera"]["Geometric Coverage-32"])
    recent_camera = support_summary(coverage["camera"]["Recent-32 + anchor"])
    geo_joint = support_summary(coverage["joint"]["Geometric Coverage-32"])
    recent_joint = support_summary(coverage["joint"]["Recent-32 + anchor"])
    report = [
        "# Geometric Coverage Eviction Visualization",
        "",
        f"- Run: `{args.run_name}`.",
        f"- Selected trajectory: row `{row_idx}` (`{item['scene']}`).",
        f"- Selected section: `{section_idx}`.",
        f"- Candidate bank before eviction: `{len(snapshot['prospective'])}` frames.",
        f"- Evicted: `{len(snapshot['evicted'])}`; retained: `{len(snapshot['retained'])}`.",
        f"- Trace/recomputed eviction overlap: `{overlap:.3f}`.",
        "",
        "## Coverage Readout",
        "",
        "Coverage is measured over every frame in the pre-eviction candidate bank. Self-support is removed. The recent baseline keeps frame 0 and the 31 newest candidates, so both banks use the same 32-frame capacity and the same anchor constraint.",
        "",
        f"- Direct camera-FOV support: Geometric mean/p10 `{geo_camera['mean']:.3f}/{geo_camera['p10']:.3f}`; Recent mean/p10 `{recent_camera['mean']:.3f}/{recent_camera['p10']:.3f}`.",
        f"- Joint pose+DINO support: Geometric mean/p10 `{geo_joint['mean']:.3f}/{geo_joint['p10']:.3f}`; Recent mean/p10 `{recent_joint['mean']:.3f}/{recent_joint['p10']:.3f}`.",
        "",
        "## What The Policy Does",
        "",
        "For every pair of memories, the policy computes `K(i,j) = 0.65 exp(-pose_distance(i,j)) + 0.35 max(DINO_cosine(i,j), 0)`. A frame receives a low retention score when at least three other memories have `K >= 0.65` and one of them is a close substitute. The lowest scores are evicted until 32 frames remain.",
        "",
        "This is a local redundancy rule, not a proof of globally optimal set coverage. The coverage plot measures whether the bank produced by that rule actually preserves candidate-view support in this recorded section.",
        "",
        "## Figures",
        "",
        "1. `01_camera_space_*.png`: physical camera path, retained memories, evictions, and substitute links.",
        "2. `02_joint_affinity_space_*.png`: MDS projection of the exact pose+DINO affinity used by the policy.",
        "3. `03_score_decomposition_*.png`: observer count and uniqueness for retained and evicted frames.",
        "4. `04_evicted_substitutes_*.png`: actual evicted frames beside the retained frames that cover them.",
        "5. `05_coverage_comparison_*.png`: support curves for Geometric Coverage-32 versus Recent-32 + anchor.",
        "6. `06_geometric_memory_timeline.png`: retained and evicted frame indices throughout the rollout.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Selected row {row_idx}: {item['scene']}")
    print(f"Selected section: {section_idx}")
    print(f"Trace/recomputed eviction overlap: {overlap:.3f}")
    print(
        "Camera-FOV support mean/p10: "
        f"Geometric={geo_camera['mean']:.3f}/{geo_camera['p10']:.3f} "
        f"Recent={recent_camera['mean']:.3f}/{recent_camera['p10']:.3f}"
    )
    print(
        "Joint support mean/p10: "
        f"Geometric={geo_joint['mean']:.3f}/{geo_joint['p10']:.3f} "
        f"Recent={recent_joint['mean']:.3f}/{recent_joint['p10']:.3f}"
    )
    for output in outputs:
        if output is not None:
            print(f"Wrote: {output}")
    print(f"Wrote: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
