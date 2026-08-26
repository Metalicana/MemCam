"""Visualize rarity-irreplaceability clustering and eviction decisions.

The completed access trace supplies the actual evicted frame identities and
their scores. For one informative section, this utility reconstructs the
candidate bank and recomputes RI descriptors from the saved video so that the
whole decision boundary, not only the discarded frames, can be visualized.
It never reruns video generation.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = REPO_ROOT / "utils" / "visualize_coverage_hysteresis.py"

COLORS = {
    "retained_old": "#2878b5",
    "retained_new": "#2b9b58",
    "evicted_old": "#a33d3d",
    "evicted_new": "#e04b3f",
    "anchor": "#7656a6",
}


def load_common_module():
    spec = importlib.util.spec_from_file_location("memcam_eviction_common", COMMON_PATH)
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


def ri_evictions_by_section(events):
    grouped = defaultdict(list)
    for row in events:
        if row.get("event") != "memory_eviction":
            continue
        if row.get("memory_policy") != "rarity_irreplaceability":
            continue
        section_idx = safe_int(row.get("section_idx"))
        frame_idx = safe_int(row.get("evicted_memory_frame"))
        if section_idx is None or frame_idx is None:
            continue
        grouped[section_idx].append(row)
    return grouped


def reconstruct_ri_snapshots(events, budget=32, frames_per_section=77):
    """Reconstruct each RI candidate pool from generation indices and evictions."""
    evictions = ri_evictions_by_section(events)
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
            "Could not reconstruct the RI bank within budget for sections "
            f"{malformed[:5]}"
        )
    return snapshots


def trace_information_score(events):
    rows = [row for values in ri_evictions_by_section(events).values() for row in values]
    cluster_sizes = [safe_float(row.get("eviction_cluster_size")) for row in rows]
    cluster_sizes = [value for value in cluster_sizes if value is not None]
    duplicates = sum(
        safe_int(row.get("eviction_rgb_nearest_frame")) is not None for row in rows
    )
    return len(rows) + duplicates + (max(cluster_sizes) if cluster_sizes else 0.0)


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
        if not ri_evictions_by_section(events):
            continue
        candidates.append(
            (trace_information_score(events), row_idx, item, trace_path, events)
        )
    if not candidates:
        suffix = "" if requested_row is None else f" for row {requested_row}"
        raise RuntimeError(f"No completed RI trace and video found{suffix}")
    return max(candidates, key=lambda row: (row[0], row[1]))


def choose_section(snapshots, requested_section=None):
    if requested_section is not None:
        section_idx = int(requested_section)
        if section_idx not in snapshots:
            raise ValueError(f"Section {section_idx} is absent from the RI trace")
        return section_idx

    max_section = max(snapshots)
    candidates = []
    for section_idx, snapshot in snapshots.items():
        if section_idx == 0:
            continue
        rows = snapshot["eviction_rows"]
        cluster_sizes = [safe_float(row.get("eviction_cluster_size")) for row in rows]
        cluster_sizes = [value for value in cluster_sizes if value is not None]
        low_scores = [safe_float(row.get("eviction_score")) for row in rows]
        low_scores = [value for value in low_scores if value is not None]
        score = max(cluster_sizes, default=0.0)
        score += 0.01 * len(rows)
        score += 0.5 * section_idx / max(max_section, 1)
        if low_scores:
            score += 1.0 / max(np.median(low_scores), 1e-8)
        candidates.append((score, section_idx))
    if not candidates:
        raise RuntimeError("RI trace has no mature section to visualize")
    return max(candidates)[1]


def select_duplicate_evictions(events, limit=5):
    candidates = []
    for section_idx, rows in ri_evictions_by_section(events).items():
        for row in rows:
            frame_idx = safe_int(row.get("evicted_memory_frame"))
            nearest = safe_int(row.get("eviction_rgb_nearest_frame"))
            score = safe_float(row.get("eviction_score"))
            cluster_size = safe_float(row.get("eviction_cluster_size"))
            if None in (frame_idx, nearest, score, cluster_size):
                continue
            candidates.append((-cluster_size, score, -section_idx, row))
    candidates.sort(key=lambda item: item[:3])

    selected = []
    used_sections = set()
    for _, _, _, row in candidates:
        section_idx = int(row["section_idx"])
        if section_idx in used_sections:
            continue
        selected.append(row)
        used_sections.add(section_idx)
        if len(selected) >= int(limit):
            break
    return selected


def frame_status(frame_idx, snapshot):
    frame_idx = int(frame_idx)
    if frame_idx == 0:
        return "anchor"
    is_new = frame_idx in set(snapshot["incoming"])
    is_evicted = frame_idx in set(snapshot["evicted"])
    if is_evicted:
        return "evicted_new" if is_new else "evicted_old"
    return "retained_new" if is_new else "retained_old"


def representative_eviction_labels(snapshot, details, limit=7):
    """Choose a few informative labels without covering dense point clouds."""
    old_frames = [
        idx for idx in snapshot["evicted"] if idx in set(snapshot["current_memory"])
    ]
    new_frames = [
        idx for idx in snapshot["evicted"] if idx in set(snapshot["incoming"])
    ]
    old_frames.sort(key=lambda idx: details[idx]["score"])

    # Pick at most one low-score example from each large incoming cluster.
    cluster_examples = {}
    for frame_idx in sorted(new_frames, key=lambda idx: details[idx]["score"]):
        cluster_id = int(details[frame_idx]["cluster_id"])
        cluster_examples.setdefault(cluster_id, frame_idx)
    new_examples = sorted(
        cluster_examples.values(),
        key=lambda idx: (-details[idx]["cluster_size"], details[idx]["score"]),
    )
    selected = old_frames[:3] + new_examples
    return selected[: int(limit)]


def render_duplicate_montage(common, rows, images, output_path):
    if not rows:
        return None
    plt = common.configure_matplotlib()
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(12.8, 3.65 * len(rows)),
        squeeze=False,
    )
    for row_idx, row in enumerate(rows):
        evicted = int(row["evicted_memory_frame"])
        nearest = int(row["eviction_rgb_nearest_frame"])
        left = common.image_array(images, evicted)
        right = common.image_array(images, nearest)
        difference = np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32)), axis=2)
        axes[row_idx, 0].imshow(left)
        axes[row_idx, 1].imshow(right)
        axes[row_idx, 2].imshow(difference, cmap="magma", vmin=0, vmax=80)
        axes[row_idx, 0].set_title(
            f"evicted f{evicted}\nRI={float(row['eviction_score']):.4f}",
            color=COLORS["evicted_new"],
            fontweight="bold",
        )
        axes[row_idx, 1].set_title(
            f"nearest substitute f{nearest}\nRGB distance={float(row['eviction_rgb_nearest_distance']):.4f}"
        )
        axes[row_idx, 2].set_title(
            f"absolute difference\ncluster size={int(row['eviction_cluster_size'])}"
        )
        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle(
        "Why RI assigns a low score: an evicted frame already has a visual substitute",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.8)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_score_landscape(common, snapshot, details, output_path):
    plt = common.configure_matplotlib()
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    for status in (
        "retained_old",
        "retained_new",
        "evicted_old",
        "evicted_new",
        "anchor",
    ):
        frames = [
            idx for idx in snapshot["prospective"] if frame_status(idx, snapshot) == status
        ]
        if not frames:
            continue
        marker = "*" if status == "anchor" else ("X" if "evicted" in status else "o")
        ax.scatter(
            [details[idx]["rarity"] for idx in frames],
            [details[idx]["irreplaceability"] for idx in frames],
            s=[min(45 + 8 * details[idx]["cluster_size"], 180) for idx in frames],
            marker=marker,
            color=COLORS[status],
            alpha=0.86,
            edgecolors="white" if marker != "X" else COLORS[status],
            linewidths=0.8,
        )
    finite_scores = [
        details[idx]["score"]
        for idx in snapshot["prospective"]
        if np.isfinite(details[idx]["score"])
    ]
    if finite_scores:
        x_max = max(details[idx]["rarity"] for idx in snapshot["prospective"])
        x = np.linspace(max(x_max * 0.03, 1e-4), x_max * 1.05, 200)
        logged_scores = [
            safe_float(row.get("eviction_score"))
            for row in snapshot.get("eviction_rows", [])
        ]
        logged_scores = [value for value in logged_scores if value is not None]
        cutoff = (
            max(logged_scores)
            if logged_scores
            else max(details[idx]["score"] for idx in snapshot["evicted"])
        )
        y_values = [details[idx]["irreplaceability"] for idx in snapshot["prospective"]]
        y_limit = max(max(y_values) * 1.3, 1e-3)
        boundary = cutoff / x
        visible = boundary <= y_limit
        ax.plot(
            x[visible],
            boundary[visible],
            linestyle="--",
            color="#333333",
            linewidth=1.5,
        )
        ax.text(
            x[-1],
            min(cutoff / x[-1], y_limit),
            " logged eviction cutoff",
            fontsize=10,
            va="bottom",
        )
        ax.set_ylim(0, y_limit)
    for frame_idx in representative_eviction_labels(snapshot, details, limit=6):
        ax.annotate(
            f"f{frame_idx}",
            (details[frame_idx]["rarity"], details[frame_idx]["irreplaceability"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("rarity: larger means a smaller DINO cluster")
    ax.set_ylabel("irreplaceability: RGB distance to nearest substitute")
    ax.set_title(
        f"RI eviction at section {snapshot['section_idx']}: score = rarity x irreplaceability",
        fontsize=16,
        fontweight="bold",
    )
    legend_handles = []
    for status in (
        "retained_old",
        "retained_new",
        "evicted_old",
        "evicted_new",
        "anchor",
    ):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=("*" if status == "anchor" else ("X" if "evicted" in status else "o")),
                color="none",
                markerfacecolor=COLORS[status],
                markeredgecolor=COLORS[status],
                markersize=9,
                label=status.replace("_", " "),
            )
        )
    ax.legend(handles=legend_handles, frameon=False, loc="best")
    ax.grid(color="#ededed", linewidth=0.8)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_cluster_map(common, snapshot, dino_features, details, output_path):
    plt = common.configure_matplotlib()
    frames = snapshot["prospective"]
    coordinates = common.pca_2d(np.stack([dino_features[idx] for idx in frames]))
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(11, 8))
    labeled_evictions = set(
        representative_eviction_labels(snapshot, details, limit=9)
    )
    for position, frame_idx in enumerate(frames):
        status = frame_status(frame_idx, snapshot)
        cluster_id = int(details[frame_idx]["cluster_id"])
        marker = "*" if status == "anchor" else ("X" if "evicted" in status else "o")
        ax.scatter(
            coordinates[position, 0],
            coordinates[position, 1],
            marker=marker,
            s=130 if "evicted" in status or status == "anchor" else 60,
            color=cmap(cluster_id % 20),
            edgecolors=COLORS[status],
            linewidths=2.0 if "evicted" in status else 1.0,
            alpha=0.9,
        )
        if frame_idx in labeled_evictions:
            ax.annotate(
                f"f{frame_idx}",
                coordinates[position],
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    handles = [
        ax.scatter([], [], color=COLORS[status], marker=("X" if "evicted" in status else "o"), s=75, label=status.replace("_", " "))
        for status in ("retained_old", "retained_new", "evicted_old", "evicted_new")
    ]
    ax.legend(handles=handles, frameon=False, loc="best")
    ax.set_xlabel("DINO PCA 1")
    ax.set_ylabel("DINO PCA 2")
    ax.set_title(
        f"Appearance clusters before RI eviction at section {snapshot['section_idx']}",
        fontsize=17,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.99,
        "Fill color: DINO cluster   Outline/marker: policy decision",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#444444",
    )
    ax.grid(color="#ededed", linewidth=0.8)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_cluster_contact_sheet(
    common,
    snapshot,
    images,
    details,
    output_path,
    max_clusters=4,
    frames_per_cluster=6,
):
    clusters = defaultdict(list)
    for frame_idx in snapshot["prospective"]:
        clusters[int(details[frame_idx]["cluster_id"])].append(frame_idx)
    selected = sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0]))[
        : int(max_clusters)
    ]
    if not selected:
        return None
    plt = common.configure_matplotlib()
    fig, axes = plt.subplots(
        len(selected),
        int(frames_per_cluster),
        figsize=(2.55 * int(frames_per_cluster), 2.7 * len(selected)),
        squeeze=False,
    )
    for row_idx, (cluster_id, members) in enumerate(selected):
        ranked = sorted(members, key=lambda idx: details[idx]["score"])
        display = ranked[: int(frames_per_cluster)]
        for col_idx, ax in enumerate(axes[row_idx]):
            if col_idx >= len(display):
                ax.axis("off")
                continue
            frame_idx = display[col_idx]
            status = frame_status(frame_idx, snapshot)
            ax.imshow(common.image_array(images, frame_idx))
            ax.set_title(
                f"f{frame_idx} | {status.replace('_', ' ')}\nRI={details[frame_idx]['score']:.4f}",
                color=COLORS[status],
                fontsize=9,
                fontweight="bold" if "evicted" in status else "normal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(COLORS[status])
                spine.set_linewidth(3)
        axes[row_idx, 0].set_ylabel(
            f"cluster {cluster_id}\nsize={len(members)}",
            fontweight="bold",
        )
    fig.suptitle(
        "Largest RI clusters: low-scoring duplicates are removed first",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
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
        old_evicted = [idx for idx in snapshot["evicted"] if idx in snapshot["current_memory"]]
        new_evicted = [idx for idx in snapshot["evicted"] if idx in snapshot["incoming"]]
        ax.scatter(
            [section_idx] * len(old_evicted),
            old_evicted,
            marker="x",
            s=22,
            color=COLORS["evicted_old"],
            alpha=0.75,
        )
        ax.scatter(
            [section_idx] * len(new_evicted),
            new_evicted,
            marker="x",
            s=22,
            color=COLORS["evicted_new"],
            alpha=0.75,
        )
    handles = [
        ax.scatter([], [], color=COLORS["retained_old"], s=35, label="retained bank"),
        ax.scatter([], [], color=COLORS["evicted_old"], marker="x", s=35, label="old memory evicted"),
        ax.scatter([], [], color=COLORS["evicted_new"], marker="x", s=35, label="new frame immediately evicted"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left")
    ax.set_xlabel("generation section")
    ax.set_ylabel("memory frame index")
    ax.set_title("RI memory evolution under a fixed 32-frame budget", fontsize=17, fontweight="bold")
    ax.grid(color="#ededed", linewidth=0.75)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
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
    parser = argparse.ArgumentParser(description="Visualize completed RI evictions.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run_name", default="ri_b32_dino_rgb_k3")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--rarity_neighbors", type=int, default=3)
    parser.add_argument("--frames_per_section", type=int, default=77)
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--section", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_duplicate_examples", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    common = load_common_module()
    manifest_rows = common.load_manifest(args.manifest, args.duration)
    trace_score, row_idx, item, trace_path, events = choose_trace(
        common,
        manifest_rows,
        args.root,
        args.run_name,
        requested_row=args.row,
    )
    snapshots = reconstruct_ri_snapshots(
        events,
        budget=args.budget,
        frames_per_section=args.frames_per_section,
    )
    section_idx = choose_section(snapshots, args.section)
    snapshot = snapshots[section_idx]
    duplicate_rows = select_duplicate_evictions(
        events,
        limit=args.max_duplicate_examples,
    )

    policy_module = common.load_policy_module()
    video_path = common.resolve_video_path(args.root, args.run_name, item)
    image_frames = set(snapshot["prospective"])
    for row in duplicate_rows:
        image_frames.add(int(row["evicted_memory_frame"]))
        image_frames.add(int(row["eviction_rgb_nearest_frame"]))
    images = common.load_video_frames(video_path, image_frames)
    missing = [idx for idx in image_frames if idx not in images]
    if missing:
        raise RuntimeError(f"Could not decode frames: {sorted(missing)[:10]}")

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
    _, details = policy_module.compute_rarity_irreplaceability_scores(
        memory_frame_indices=snapshot["prospective"],
        pinned_frames={0},
        rarity_neighbors=args.rarity_neighbors,
        dino_features=dino_features,
        rgb_features=rgb_features,
        return_details=True,
    )

    traced_evicted = set(snapshot["evicted"])
    recomputed_evicted = {
        idx
        for idx, _ in sorted(
            (
                (idx, row["score"])
                for idx, row in details.items()
                if idx not in {0, snapshot["section_end"]}
            ),
            key=lambda item: (item[1], item[0]),
        )[: max(0, len(snapshot["prospective"]) - args.budget)]
    }
    overlap = (
        len(traced_evicted & recomputed_evicted) / len(traced_evicted)
        if traced_evicted
        else 1.0
    )

    figures = args.output_dir / "figures"
    tables = args.output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    outputs = [
        render_duplicate_montage(
            common,
            duplicate_rows,
            images,
            figures / "01_low_score_duplicates.png",
        ),
        render_score_landscape(
            common,
            snapshot,
            details,
            figures / f"02_ri_score_landscape_section_{section_idx}.png",
        ),
        render_cluster_map(
            common,
            snapshot,
            dino_features,
            details,
            figures / f"03_dino_clusters_section_{section_idx}.png",
        ),
        render_cluster_contact_sheet(
            common,
            snapshot,
            images,
            details,
            figures / f"04_cluster_contact_sheet_section_{section_idx}.png",
        ),
        render_timeline(
            common,
            snapshots,
            figures / "05_ri_memory_timeline.png",
        ),
    ]

    decision_rows = []
    for frame_idx in snapshot["prospective"]:
        decision_rows.append(
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
    write_csv(tables / "ri_section_scores.csv", decision_rows)
    write_csv(tables / "low_score_duplicate_examples.csv", duplicate_rows)

    report = [
        "# RI Eviction Visualization",
        "",
        f"- Run: `{args.run_name}`.",
        f"- Selected trajectory: row `{row_idx}` (`{item['scene']}`).",
        f"- Selected section: `{section_idx}`.",
        f"- Candidate bank before eviction: `{len(snapshot['prospective'])}` frames.",
        f"- Evicted: `{len(snapshot['evicted'])}`; retained: `{len(snapshot['retained'])}`.",
        f"- Trace/recomputed eviction overlap: `{overlap:.3f}`.",
        "",
        "Actual eviction identities and their logged RI values come from the generation trace. Whole-bank clusters and scores are recomputed with the same RI implementation from the saved MP4. Small differences are possible because the saved video has undergone MP4 compression.",
        "",
        "## Figures",
        "",
        "1. `01_low_score_duplicates.png`: each actual eviction beside its nearest RGB substitute.",
        "2. `02_ri_score_landscape_*.png`: rarity versus irreplaceability, with the product-score eviction boundary.",
        "3. `03_dino_clusters_*.png`: the DINO appearance clusters and actual eviction markers.",
        "4. `04_cluster_contact_sheet_*.png`: image members of the largest clusters.",
        "5. `05_ri_memory_timeline.png`: retained and evicted frames over the rollout.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Selected row {row_idx}: {item['scene']}")
    print(f"Selected section: {section_idx}")
    print(f"Trace/recomputed eviction overlap: {overlap:.3f}")
    for output in outputs:
        if output is not None:
            print(f"Wrote: {output}")
    print(f"Wrote: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
