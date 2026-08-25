"""Visualize Coverage-Hysteresis admission, clustering, and eviction decisions.

The utility reconstructs a policy update from its access trace, extracts the
corresponding generated frames, and recomputes the exact DINO/RGB RI clusters
and 0.75 Geo + 0.25 RI retention scores. It does not rerun video generation.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_MODULE_PATH = REPO_ROOT / "diffsynth" / "pipelines" / "memory_policies.py"

STATUS_COLORS = {
    "retained": "#2878b5",
    "admitted": "#2b9b58",
    "evicted": "#d64242",
    "rejected": "#ef8a24",
    "rolling": "#7656a6",
}


def load_policy_module():
    spec = importlib.util.spec_from_file_location(
        "memcam_visualization_memory_policies",
        POLICY_MODULE_PATH,
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


def as_bool(value):
    if value in (True, 1, "1", "true", "True"):
        return True
    if value in (False, 0, "0", "false", "False"):
        return False
    return False


def unique_ints(values):
    return list(dict.fromkeys(int(value) for value in values if value is not None))


def load_manifest(path, duration):
    rows = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            if int(item["duration_sec"]) != int(duration):
                continue
            item["_row"] = row_idx
            rows[row_idx] = item
    return rows


def load_trace(path):
    events = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return events


def trace_groups(events):
    admissions = defaultdict(list)
    evictions = defaultdict(list)
    updates = {}
    for row in events:
        section_idx = safe_int(row.get("section_idx"))
        if section_idx is None:
            continue
        event = row.get("event")
        if event == "coverage_hysteresis_admission":
            admissions[section_idx].append(row)
        elif event == "memory_eviction":
            evictions[section_idx].append(row)
        elif event == "coverage_hysteresis_update":
            updates[section_idx] = row
    return admissions, evictions, updates


def reconstruct_section(events, section_idx):
    admissions, evictions, updates = trace_groups(events)
    section_idx = int(section_idx)
    if section_idx not in updates:
        raise ValueError(f"No coverage_hysteresis_update for section {section_idx}")

    previous_sections = [idx for idx in updates if idx < section_idx]
    if previous_sections:
        previous = updates[max(previous_sections)]
        current_memory = unique_ints(previous.get("retained_memory_frames", []))
        rolling = safe_int(previous.get("rolling_only_frame"))
        if rolling is not None:
            current_memory = [idx for idx in current_memory if idx != rolling]
    else:
        current_memory = [0]

    admission_rows = sorted(
        admissions.get(section_idx, []),
        key=lambda row: safe_int(row.get("candidate_memory_frame")) or -1,
    )
    admitted = unique_ints(
        row.get("candidate_memory_frame")
        for row in admission_rows
        if as_bool(row.get("hysteresis_admitted"))
    )
    rejected = unique_ints(
        row.get("candidate_memory_frame")
        for row in admission_rows
        if not as_bool(row.get("hysteresis_admitted"))
    )
    prospective = unique_ints(current_memory + admitted)
    retained = unique_ints(updates[section_idx].get("retained_memory_frames", []))
    evicted = unique_ints(
        row.get("evicted_memory_frame") for row in evictions.get(section_idx, [])
    )
    rolling = safe_int(updates[section_idx].get("rolling_only_frame"))

    return {
        "section_idx": section_idx,
        "current_memory": current_memory,
        "admission_rows": admission_rows,
        "admitted": admitted,
        "rejected": rejected,
        "prospective": prospective,
        "retained": retained,
        "evicted": evicted,
        "rolling_only_frame": rolling,
        "update": updates[section_idx],
        "eviction_rows": evictions.get(section_idx, []),
    }


def trace_information_score(events):
    admissions, evictions, updates = trace_groups(events)
    rejected = [
        row
        for rows in admissions.values()
        for row in rows
        if not as_bool(row.get("hysteresis_admitted"))
    ]
    similarities = [
        safe_float(row.get("hysteresis_max_view_similarity")) for row in rejected
    ]
    similarities = [value for value in similarities if value is not None]
    return (
        len(rejected)
        + 3.0 * sum(len(rows) for rows in evictions.values())
        + 10.0 * (float(np.mean(similarities)) if similarities else 0.0)
        + 0.01 * len(updates)
    )


def find_trace(root, run_name, item):
    trace_dir = Path(root) / run_name / "access_traces"
    exact = trace_dir / f"{item['output_prefix']}custom.jsonl"
    if exact.is_file():
        return exact
    matches = sorted(trace_dir.glob(f"{item['output_prefix']}*.jsonl"))
    return matches[0] if matches else None


def choose_trace(manifest_rows, root, run_name, requested_row=None):
    candidates = []
    for row_idx, item in manifest_rows.items():
        if requested_row is not None and int(row_idx) != int(requested_row):
            continue
        path = find_trace(root, run_name, item)
        if path is None:
            continue
        events = load_trace(path)
        _, _, updates = trace_groups(events)
        video_path = Path(root) / run_name / f"{item['output_prefix']}custom.mp4"
        if not updates or not video_path.is_file():
            continue
        score = trace_information_score(events)
        candidates.append((score, row_idx, item, path, events))
    if not candidates:
        suffix = "" if requested_row is None else f" for row {requested_row}"
        raise RuntimeError(f"No Coverage-Hysteresis trace found{suffix}")
    return max(candidates, key=lambda row: (row[0], row[1]))


def choose_cluster_section(events, requested_section=None):
    admissions, evictions, updates = trace_groups(events)
    if requested_section is not None:
        section_idx = int(requested_section)
        if section_idx not in updates:
            raise ValueError(f"Section {section_idx} is absent from the trace")
        return section_idx

    max_section = max(updates) if updates else 1
    scored = []
    for section_idx in sorted(updates):
        if section_idx == 0:
            continue
        rejected = sum(
            not as_bool(row.get("hysteresis_admitted"))
            for row in admissions.get(section_idx, [])
        )
        evicted = len(evictions.get(section_idx, []))
        admitted = sum(
            as_bool(row.get("hysteresis_admitted"))
            for row in admissions.get(section_idx, [])
        )
        score = 2.0 * rejected + 4.0 * evicted + admitted
        score += 2.0 * section_idx / max(max_section, 1)
        scored.append((score, section_idx))
    if not scored:
        raise RuntimeError("Trace has no visualizable policy updates")
    return max(scored)[1]


def select_duplicate_rows(events, limit):
    admissions, _, _ = trace_groups(events)
    candidates = []
    for section_idx, rows in admissions.items():
        for row in rows:
            if as_bool(row.get("hysteresis_admitted")):
                continue
            candidate = safe_int(row.get("candidate_memory_frame"))
            incumbent = safe_int(row.get("hysteresis_nearest_reference_frame"))
            similarity = safe_float(row.get("hysteresis_max_view_similarity"))
            if None in (candidate, incumbent, similarity):
                continue
            candidates.append((similarity, section_idx, candidate, row))
    candidates.sort(reverse=True, key=lambda item: (item[0], item[1], item[2]))

    selected = []
    used_sections = Counter()
    for _, section_idx, _, row in candidates:
        if used_sections[section_idx] >= 1:
            continue
        selected.append(row)
        used_sections[section_idx] += 1
        if len(selected) >= int(limit):
            break
    return selected


def select_eviction_rows(events, limit):
    _, evictions, _ = trace_groups(events)
    candidates = []
    for section_idx, rows in evictions.items():
        for row in rows:
            score = safe_float(row.get("eviction_score"))
            geo = safe_float(row.get("eviction_slamri_slam_norm"))
            ri = safe_float(row.get("eviction_slamri_ri_norm"))
            if None in (score, geo, ri):
                continue
            candidates.append((score, -section_idx, row))
    candidates.sort(key=lambda item: (item[0], item[1]))

    selected = []
    used_sections = Counter()
    for _, _, row in candidates:
        section_idx = int(row["section_idx"])
        if used_sections[section_idx] >= 1:
            continue
        selected.append(row)
        used_sections[section_idx] += 1
        if len(selected) >= int(limit):
            break
    return selected


def resolve_video_path(root, run_name, item):
    path = Path(root) / run_name / f"{item['output_prefix']}custom.mp4"
    if not path.is_file():
        raise FileNotFoundError(f"Missing generated video: {path}")
    return path


def load_video_frames(video_path, frame_indices):
    import imageio.v2 as imageio

    images = {}
    reader = imageio.get_reader(str(video_path))
    try:
        for frame_idx in sorted(set(int(idx) for idx in frame_indices)):
            try:
                array = reader.get_data(frame_idx)
            except Exception as exc:
                print(f"[warn] failed to read frame {frame_idx}: {exc}")
                continue
            images[frame_idx] = Image.fromarray(
                np.asarray(array).astype(np.uint8)
            ).convert("RGB")
    finally:
        reader.close()
    return images


def nearest_retained_frame(policy_module, c2ws, evicted_frame, retained_frames):
    retained_frames = [
        int(frame_idx)
        for frame_idx in retained_frames
        if int(frame_idx) != int(evicted_frame)
    ]
    if not retained_frames:
        return None, None
    similarities = policy_module.camera_trajectory_similarity(
        c2ws=c2ws,
        query_frame_indices=[int(evicted_frame)],
        memory_frame_indices=retained_frames,
    )[0]
    position = int(np.argmax(similarities))
    return retained_frames[position], float(similarities[position])


def pca_2d(features):
    features = np.asarray(features, dtype=np.float64)
    if len(features) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    centered = features - np.mean(features, axis=0, keepdims=True)
    if len(features) == 1:
        return np.zeros((1, 2), dtype=np.float64)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[: min(2, len(vh))].T
    coordinates = centered @ components
    if coordinates.shape[1] == 1:
        coordinates = np.concatenate(
            [coordinates, np.zeros((len(coordinates), 1))], axis=1
        )
    return coordinates[:, :2]


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_eviction_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def image_array(images, frame_idx):
    image = images.get(int(frame_idx))
    if image is None:
        return np.full((180, 320, 3), 230, dtype=np.uint8)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def render_duplicate_montage(rows, images, output_path, threshold):
    if not rows:
        return None
    plt = configure_matplotlib()
    fig, axes = plt.subplots(len(rows), 3, figsize=(12.5, 3.1 * len(rows)))
    axes = np.asarray(axes).reshape(len(rows), 3)
    for row_idx, row in enumerate(rows):
        incumbent = int(row["hysteresis_nearest_reference_frame"])
        candidate = int(row["candidate_memory_frame"])
        similarity = float(row["hysteresis_max_view_similarity"])
        old = image_array(images, incumbent)
        new = image_array(images, candidate)
        if old.shape != new.shape:
            new = np.asarray(
                Image.fromarray(new).resize((old.shape[1], old.shape[0]))
            )
        difference = np.mean(
            np.abs(old.astype(np.float32) - new.astype(np.float32)), axis=2
        )

        axes[row_idx, 0].imshow(old)
        axes[row_idx, 0].set_title(f"Incumbent kept | frame {incumbent}")
        axes[row_idx, 1].imshow(new)
        axes[row_idx, 1].set_title(f"Incoming rewrite rejected | frame {candidate}")
        axes[row_idx, 2].imshow(difference, cmap="magma", vmin=0, vmax=80)
        axes[row_idx, 2].set_title(
            f"Same-view match: {similarity:.3f} >= {threshold:.2f}"
        )
        for ax in axes[row_idx]:
            ax.axis("off")
        axes[row_idx, 0].set_ylabel(
            f"section {int(row['section_idx'])}",
            fontsize=11,
            fontweight="bold",
        )
    fig.suptitle(
        "Coverage hysteresis blocks redundant autoregressive rewrites",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_eviction_scorecards(
    rows,
    images,
    nearest_survivors,
    output_path,
    geometric_weight,
):
    if not rows:
        return None
    plt = configure_matplotlib()
    fig = plt.figure(figsize=(13.5, 3.2 * len(rows)))
    grid = fig.add_gridspec(len(rows), 3, width_ratios=(1.25, 1.25, 1.7))
    for row_idx, row in enumerate(rows):
        evicted = int(row["evicted_memory_frame"])
        survivor, view_similarity = nearest_survivors[(int(row["section_idx"]), evicted)]
        score = float(row["eviction_score"])
        geo = float(row["eviction_slamri_slam_norm"])
        ri = float(row["eviction_slamri_ri_norm"])
        geo_contribution = geometric_weight * geo
        ri_contribution = (1.0 - geometric_weight) * ri

        ax_old = fig.add_subplot(grid[row_idx, 0])
        ax_old.imshow(image_array(images, evicted))
        ax_old.set_title(f"Evicted frame {evicted}", color=STATUS_COLORS["evicted"])
        ax_old.axis("off")

        ax_peer = fig.add_subplot(grid[row_idx, 1])
        if survivor is None:
            ax_peer.imshow(np.full((180, 320, 3), 230, dtype=np.uint8))
            ax_peer.set_title("No surviving peer")
        else:
            ax_peer.imshow(image_array(images, survivor))
            ax_peer.set_title(
                f"Closest retained view {survivor}\nK_view={view_similarity:.3f}"
            )
        ax_peer.axis("off")

        ax_score = fig.add_subplot(grid[row_idx, 2])
        ax_score.barh(
            ["Final retention", "Weighted parts"],
            [score, geo_contribution],
            color=[STATUS_COLORS["evicted"], "#2878b5"],
        )
        ax_score.barh(
            ["Final retention", "Weighted parts"],
            [0.0, ri_contribution],
            left=[0.0, geo_contribution],
            color=["#ffffff", "#ef8a24"],
        )
        ax_score.set_xlim(0, 1.02)
        ax_score.set_xlabel("normalized retention utility")
        ax_score.set_title(
            f"section {int(row['section_idx'])}: "
            f"Geo={geo:.3f}, RI={ri:.3f}, final={score:.3f}"
        )
        ax_score.grid(axis="x", color="#e8e8e8", linewidth=0.8)
        for spine_name in ("top", "right"):
            ax_score.spines[spine_name].set_visible(False)
        ax_score.legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, color="#2878b5", label="0.75 Geo"),
                plt.Rectangle((0, 0), 1, 1, color="#ef8a24", label="0.25 RI"),
            ],
            loc="lower right",
            frameon=False,
        )
    fig.suptitle(
        "Why did this memory receive the lowest retention score?",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return output_path


def frame_status(frame_idx, snapshot):
    if frame_idx in snapshot["rejected"]:
        return "rejected"
    if frame_idx in snapshot["evicted"]:
        return "evicted"
    if frame_idx == snapshot["rolling_only_frame"]:
        return "rolling"
    if frame_idx in snapshot["admitted"]:
        return "admitted"
    return "retained"


def render_cluster_map(
    snapshot,
    frame_indices,
    dino_features,
    score_details,
    output_path,
    geometric_weight,
):
    plt = configure_matplotlib()
    coordinates = pca_2d(np.stack([dino_features[idx] for idx in frame_indices]))
    prospective = set(snapshot["prospective"])
    cluster_ids = {
        idx: int(score_details[idx]["slamri_ri_cluster_id"])
        for idx in snapshot["prospective"]
    }
    for idx in snapshot["rejected"]:
        if idx not in dino_features or not prospective:
            continue
        candidates = list(snapshot["prospective"])
        similarities = np.asarray(
            [np.dot(dino_features[idx], dino_features[candidate]) for candidate in candidates]
        )
        cluster_ids[idx] = cluster_ids[candidates[int(np.argmax(similarities))]]

    fig, (ax_map, ax_scores) = plt.subplots(
        1,
        2,
        figsize=(15.5, 7.8),
        gridspec_kw={"width_ratios": [1.05, 1.15]},
    )
    cmap = plt.get_cmap("tab20")
    markers = {
        "retained": "o",
        "admitted": "^",
        "evicted": "X",
        "rejected": "x",
        "rolling": "s",
    }
    for position, frame_idx in enumerate(frame_indices):
        status = frame_status(frame_idx, snapshot)
        cluster_id = cluster_ids.get(frame_idx, -1)
        color = "#777777" if cluster_id < 0 else cmap(cluster_id % 20)
        scatter_kwargs = {
            "marker": markers[status],
            "s": 115 if status in {"evicted", "rejected"} else 72,
            "color": color,
            "linewidth": 2.2 if status != "rejected" else 2.8,
            "alpha": 0.92,
        }
        if status != "rejected":
            scatter_kwargs["edgecolor"] = STATUS_COLORS[status]
        ax_map.scatter(
            coordinates[position, 0],
            coordinates[position, 1],
            **scatter_kwargs,
        )
        if status in {"evicted", "rejected"}:
            ax_map.annotate(
                str(frame_idx),
                coordinates[position],
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    legend_handles = [
        ax_map.scatter([], [], marker=markers[status], color=STATUS_COLORS[status], s=75, label=status)
        for status in ("retained", "admitted", "evicted", "rejected", "rolling")
    ]
    ax_map.legend(handles=legend_handles, frameon=False, loc="best")
    ax_map.set_title("DINO appearance clusters", fontweight="bold")
    ax_map.set_xlabel("DINO PCA 1")
    ax_map.set_ylabel("DINO PCA 2")
    ax_map.grid(color="#ededed", linewidth=0.8)
    for spine_name in ("top", "right"):
        ax_map.spines[spine_name].set_visible(False)

    ranked = sorted(
        snapshot["prospective"],
        key=lambda idx: float(score_details[idx]["score"]),
    )[: min(16, len(snapshot["prospective"]))]
    labels = [f"f{idx}  c{cluster_ids[idx]}" for idx in ranked]
    geo_parts = [
        geometric_weight * float(score_details[idx]["slamri_slam_norm"])
        for idx in ranked
    ]
    ri_parts = [
        (1.0 - geometric_weight) * float(score_details[idx]["slamri_ri_norm"])
        for idx in ranked
    ]
    positions = np.arange(len(ranked))
    ax_scores.barh(positions, geo_parts, color="#2878b5", label="0.75 Geo")
    ax_scores.barh(
        positions,
        ri_parts,
        left=geo_parts,
        color="#ef8a24",
        label="0.25 RI",
    )
    for position, frame_idx in enumerate(ranked):
        if frame_idx in snapshot["evicted"]:
            ax_scores.scatter(
                sum((geo_parts[position], ri_parts[position])) + 0.015,
                position,
                marker="x",
                color=STATUS_COLORS["evicted"],
                s=55,
                linewidth=2,
            )
    ax_scores.set_yticks(positions)
    ax_scores.set_yticklabels(labels)
    ax_scores.invert_yaxis()
    ax_scores.set_xlim(0, 1.03)
    ax_scores.set_xlabel("retention score")
    ax_scores.set_title("Lowest-scoring memories at the decision boundary", fontweight="bold")
    ax_scores.legend(frameon=False, loc="lower right")
    ax_scores.grid(axis="x", color="#ededed", linewidth=0.8)
    for spine_name in ("top", "right"):
        ax_scores.spines[spine_name].set_visible(False)

    fig.suptitle(
        f"Bank update at section {snapshot['section_idx']}: clusters and eviction utility",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_cluster_contact_sheet(
    snapshot,
    images,
    score_details,
    output_path,
    max_clusters=4,
    frames_per_cluster=6,
):
    clusters = defaultdict(list)
    for frame_idx in snapshot["prospective"]:
        cluster_id = int(score_details[frame_idx]["slamri_ri_cluster_id"])
        clusters[cluster_id].append(frame_idx)
    selected_clusters = sorted(
        clusters.items(), key=lambda item: (-len(item[1]), item[0])
    )[: int(max_clusters)]
    if not selected_clusters:
        return None

    plt = configure_matplotlib()
    fig, axes = plt.subplots(
        len(selected_clusters),
        frames_per_cluster,
        figsize=(2.5 * frames_per_cluster, 2.6 * len(selected_clusters)),
        squeeze=False,
    )
    for row_idx, (cluster_id, members) in enumerate(selected_clusters):
        members = sorted(
            members,
            key=lambda idx: float(score_details[idx]["score"]),
        )[:frames_per_cluster]
        for col_idx, ax in enumerate(axes[row_idx]):
            if col_idx >= len(members):
                ax.axis("off")
                continue
            frame_idx = members[col_idx]
            status = frame_status(frame_idx, snapshot)
            score = float(score_details[frame_idx]["score"])
            ax.imshow(image_array(images, frame_idx))
            ax.set_title(
                f"f{frame_idx} | {status}\nscore={score:.3f}",
                color=STATUS_COLORS[status],
                fontsize=10,
                fontweight="bold" if status == "evicted" else "normal",
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(STATUS_COLORS[status])
                spine.set_linewidth(3)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row_idx, 0].set_ylabel(
            f"DINO cluster {cluster_id}\nsize={len(clusters[cluster_id])}",
            fontsize=11,
            fontweight="bold",
        )
    fig.suptitle(
        "Largest RI appearance clusters: redundant memories compete within a fixed bank",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_memory_timeline(events, output_path):
    admissions, evictions, updates = trace_groups(events)
    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=(14, 7))
    for section_idx, update in sorted(updates.items()):
        retained = unique_ints(update.get("retained_memory_frames", []))
        ax.scatter(
            [section_idx] * len(retained),
            retained,
            s=8,
            color=STATUS_COLORS["retained"],
            alpha=0.42,
            linewidths=0,
        )
        rolling = safe_int(update.get("rolling_only_frame"))
        if rolling is not None:
            ax.scatter(
                section_idx,
                rolling,
                marker="s",
                s=30,
                color=STATUS_COLORS["rolling"],
            )
    for section_idx, rows in admissions.items():
        rejected = [
            int(row["candidate_memory_frame"])
            for row in rows
            if not as_bool(row.get("hysteresis_admitted"))
        ]
        ax.scatter(
            [section_idx] * len(rejected),
            rejected,
            marker="x",
            s=20,
            color=STATUS_COLORS["rejected"],
            alpha=0.7,
        )
    for section_idx, rows in evictions.items():
        frames = [int(row["evicted_memory_frame"]) for row in rows]
        ax.scatter(
            [section_idx] * len(frames),
            frames,
            marker="v",
            s=32,
            color=STATUS_COLORS["evicted"],
            alpha=0.9,
        )
    handles = [
        ax.scatter([], [], s=35, marker="o", color=STATUS_COLORS["retained"], label="retained bank"),
        ax.scatter([], [], s=35, marker="x", color=STATUS_COLORS["rejected"], label="duplicate rejected"),
        ax.scatter([], [], s=35, marker="v", color=STATUS_COLORS["evicted"], label="evicted after scoring"),
        ax.scatter([], [], s=35, marker="s", color=STATUS_COLORS["rolling"], label="one-step rolling anchor"),
    ]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="upper left")
    ax.set_xlabel("generation section")
    ax.set_ylabel("memory frame index")
    ax.set_title(
        "Streaming memory evolution under a fixed budget",
        fontsize=17,
        fontweight="bold",
    )
    ax.grid(color="#ededed", linewidth=0.75)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Coverage-Hysteresis duplicate rejection and eviction."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--run_name",
        default="coverage_hysteresis_b32_t0p90_s75_r25_k3",
    )
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--cluster_section", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_duplicate_examples", type=int, default=5)
    parser.add_argument("--max_eviction_examples", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    if args.max_duplicate_examples < 1 or args.max_eviction_examples < 1:
        raise ValueError("Example limits must be positive")

    manifest_rows = load_manifest(args.manifest, args.duration)
    if not manifest_rows:
        raise RuntimeError(f"No {args.duration}s rows found in {args.manifest}")
    trace_score, row_idx, item, trace_path, events = choose_trace(
        manifest_rows,
        args.root,
        args.run_name,
        requested_row=args.row,
    )
    section_idx = choose_cluster_section(events, args.cluster_section)
    snapshot = reconstruct_section(events, section_idx)
    duplicate_rows = select_duplicate_rows(events, args.max_duplicate_examples)
    eviction_rows = select_eviction_rows(events, args.max_eviction_examples)

    policy_module = load_policy_module()
    from dataset.poses import load_c2ws_from_json

    c2ws = load_c2ws_from_json(
        item["pose_path"],
        start_frame=int(item["start_frame"]),
        num_frames=int(item["num_frames"]),
    )
    nearest_survivors = {}
    image_frames = set(snapshot["prospective"] + snapshot["rejected"])
    for row in duplicate_rows:
        image_frames.add(int(row["candidate_memory_frame"]))
        image_frames.add(int(row["hysteresis_nearest_reference_frame"]))
    for row in eviction_rows:
        evicted = int(row["evicted_memory_frame"])
        event_snapshot = reconstruct_section(events, int(row["section_idx"]))
        survivor, similarity = nearest_retained_frame(
            policy_module,
            c2ws,
            evicted,
            event_snapshot["retained"],
        )
        nearest_survivors[(int(row["section_idx"]), evicted)] = (
            survivor,
            similarity,
        )
        image_frames.add(evicted)
        if survivor is not None:
            image_frames.add(survivor)

    video_path = resolve_video_path(args.root, args.run_name, item)
    images = load_video_frames(video_path, image_frames)
    missing_cluster_images = [
        frame_idx
        for frame_idx in snapshot["prospective"] + snapshot["rejected"]
        if frame_idx not in images
    ]
    if missing_cluster_images:
        raise RuntimeError(
            f"Could not decode cluster frames: {missing_cluster_images[:10]}"
        )

    feature_frames = unique_ints(snapshot["prospective"] + snapshot["rejected"])
    print(
        f"Encoding {len(feature_frames)} frames with DINO on {args.device} "
        f"for section {section_idx}"
    )
    extractor = policy_module.VisualMemoryFeatureExtractor(device=args.device)
    dino_batch, rgb_batch = extractor.encode_pil_images(
        [images[frame_idx] for frame_idx in feature_frames]
    )
    dino_features = {
        frame_idx: dino_batch[position]
        for position, frame_idx in enumerate(feature_frames)
    }
    rgb_features = {
        frame_idx: rgb_batch[position]
        for position, frame_idx in enumerate(feature_frames)
    }
    geometric_weight = float(
        snapshot["update"].get("geometric_coverage_weight", 0.75)
    )
    rarity_neighbors = int(snapshot["update"].get("rarity_neighbors", 3))
    _, score_details = policy_module.compute_slam_ri_blend_scores(
        memory_frame_indices=snapshot["prospective"],
        c2ws=c2ws,
        forced_keep_frames={0} & set(snapshot["prospective"]),
        dino_features=dino_features,
        rgb_features=rgb_features,
        beta=geometric_weight,
        ri_kwargs={"rarity_neighbors": rarity_neighbors},
        return_details=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    duplicate_path = render_duplicate_montage(
        duplicate_rows,
        images,
        figures_dir / "01_duplicate_rejections.png",
        threshold=float(snapshot["update"].get("view_similarity_threshold", 0.90)),
    )
    eviction_path = render_eviction_scorecards(
        eviction_rows,
        images,
        nearest_survivors,
        figures_dir / "02_eviction_scorecards.png",
        geometric_weight=geometric_weight,
    )
    cluster_frames = unique_ints(snapshot["prospective"] + snapshot["rejected"])
    cluster_map_path = render_cluster_map(
        snapshot,
        cluster_frames,
        dino_features,
        score_details,
        figures_dir / f"03_cluster_map_section_{section_idx}.png",
        geometric_weight=geometric_weight,
    )
    contact_sheet_path = render_cluster_contact_sheet(
        snapshot,
        images,
        score_details,
        figures_dir / f"04_cluster_contact_sheet_section_{section_idx}.png",
    )
    timeline_path = render_memory_timeline(
        events,
        figures_dir / "05_memory_timeline.png",
    )

    decision_rows = []
    for frame_idx in snapshot["prospective"]:
        row = dict(score_details[frame_idx])
        row.update(
            {
                "row": row_idx,
                "scene": item["scene"],
                "section_idx": section_idx,
                "memory_frame": frame_idx,
                "status": frame_status(frame_idx, snapshot),
            }
        )
        decision_rows.append(row)
    write_csv(tables_dir / "cluster_decision_scores.csv", decision_rows)
    write_csv(tables_dir / "duplicate_examples.csv", duplicate_rows)
    write_csv(tables_dir / "eviction_examples.csv", eviction_rows)

    report = [
        "# Coverage-Hysteresis Eviction Visualization",
        "",
        f"- Run: `{args.run_name}`.",
        f"- Automatically selected trajectory row: `{row_idx}` (`{item['scene']}`).",
        f"- Trace information score: `{trace_score:.3f}`.",
        f"- Cluster decision section: `{section_idx}`.",
        f"- Bank before admission: `{len(snapshot['current_memory'])}` frames.",
        f"- Novel candidates admitted: `{len(snapshot['admitted'])}`.",
        f"- Covered candidates rejected: `{len(snapshot['rejected'])}`.",
        f"- Frames evicted after scoring: `{len(snapshot['evicted'])}`.",
        f"- Retained bank after update: `{len(snapshot['retained'])}` frames.",
        "",
        "## Reading the figures",
        "",
        "1. `01_duplicate_rejections.png` shows the exact incumbent/candidate pairs used by the online hysteresis gate.",
        "2. `02_eviction_scorecards.png` decomposes each low score into its geometric and RI contributions and shows the closest surviving camera view.",
        "3. `03_cluster_map_*.png` projects DINO descriptors and marks retained, admitted, evicted, rejected, and rolling memories.",
        "4. `04_cluster_contact_sheet_*.png` shows the largest actual RI appearance clusters as images.",
        "5. `05_memory_timeline.png` shows the full streaming process under the fixed budget.",
        "",
        "DINO and RGB features are recomputed from the completed generated video using the same feature extractor and policy functions as generation. No video is regenerated.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Selected row {row_idx}: {item['scene']}")
    print(f"Selected cluster section: {section_idx}")
    for path in (
        duplicate_path,
        eviction_path,
        cluster_map_path,
        contact_sheet_path,
        timeline_path,
    ):
        if path is not None:
            print(f"Wrote: {path}")
    print(f"Wrote: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
