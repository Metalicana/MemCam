import math
from collections import OrderedDict

import numpy as np


SUPPORTED_MEMORY_POLICIES = ("unbounded", "fifo", "rarity_irreplaceability")
BUDGETED_MEMORY_POLICIES = ("fifo", "rarity_irreplaceability")


class FrameMemoryBuffer:
    def __init__(self, policy="unbounded", budget=None, pinned_frames=None):
        if policy not in SUPPORTED_MEMORY_POLICIES:
            raise ValueError(
                f"Unsupported memory policy '{policy}'. "
                f"Expected one of {SUPPORTED_MEMORY_POLICIES}."
            )
        if policy in BUDGETED_MEMORY_POLICIES and budget is None:
            raise ValueError(f"{policy} memory policy requires an explicit memory budget")
        if budget is not None and budget <= 0:
            raise ValueError("memory budget must be positive when provided")

        self.policy = policy
        self.budget = budget
        self._frames = OrderedDict()
        self._stats = {}
        self._next_order = 0
        self._pinned_frames = set(pinned_frames or [])

    def add(self, frame_idx, evict=True, eviction_scores=None, protected_frames=None):
        if frame_idx not in self._frames:
            self._stats[frame_idx] = {
                "insert_order": self._next_order,
                "selected_count": 0,
                "selection_overlap_sum": 0.0,
                "best_selection_overlap": 0.0,
                "score": 0.0,
            }
            self._next_order += 1
        self._frames[frame_idx] = None
        if eviction_scores:
            self.set_scores(eviction_scores)
        if evict:
            return self.evict_to_budget(protected_frames=protected_frames)
        return []

    def update(self, frame_indices, eviction_scores=None, protected_frames=None):
        evicted = []
        for frame_idx in frame_indices:
            evicted.extend(self.add(frame_idx, evict=False))
        if eviction_scores:
            self.set_scores(eviction_scores)
        evicted.extend(self.evict_to_budget(protected_frames=protected_frames))
        return evicted

    def set_scores(self, scores):
        for frame_idx, score in scores.items():
            if frame_idx in self._stats:
                self._stats[frame_idx]["score"] = float(score)

    def record_selection(self, frame_idx, overlap):
        if frame_idx not in self._stats:
            return
        stats = self._stats[frame_idx]
        stats["selected_count"] += 1
        stats["selection_overlap_sum"] += max(float(overlap), 0.0)
        stats["best_selection_overlap"] = max(
            stats["best_selection_overlap"],
            max(float(overlap), 0.0),
        )

    def evict_to_budget(self, protected_frames=None):
        if self.budget is None or self.policy == "unbounded":
            return []

        protected_frames = set(protected_frames or []) | self._pinned_frames
        evicted = []
        while len(self._frames) > self.budget:
            evictable = [
                frame_idx
                for frame_idx in self._frames.keys()
                if frame_idx not in protected_frames
            ]
            if not evictable:
                break

            if self.policy == "fifo":
                evicted_frame_idx = evictable[0]
            else:
                evicted_frame_idx = min(
                    evictable,
                    key=lambda idx: (
                        self._stats[idx].get("score", 0.0),
                        self._stats[idx]["insert_order"],
                    ),
                )

            self._frames.pop(evicted_frame_idx, None)
            self._stats.pop(evicted_frame_idx, None)
            evicted.append(evicted_frame_idx)
        return evicted

    def candidates(self, exclude_frames=None):
        exclude_frames = set(exclude_frames or [])
        return [frame_idx for frame_idx in self._frames.keys() if frame_idx not in exclude_frames]

    def selected_count(self, frame_idx):
        return self._stats.get(frame_idx, {}).get("selected_count", 0)

    def mean_selection_overlap(self, frame_idx):
        stats = self._stats.get(frame_idx)
        if not stats or stats["selected_count"] == 0:
            return 0.0
        return stats["selection_overlap_sum"] / stats["selected_count"]

    def __len__(self):
        return len(self._frames)


def rotation_distance(rotation_a, rotation_b):
    relative = rotation_a.T @ rotation_b
    cosine = (np.trace(relative) - 1.0) / 2.0
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.acos(cosine) / math.pi


def pose_distances(c2ws, frame_indices, target_indices, rotation_weight=2.0):
    frame_indices = list(frame_indices)
    target_indices = list(target_indices)
    if not frame_indices or not target_indices:
        return np.zeros((len(frame_indices), len(target_indices)), dtype=np.float64)

    frame_positions = c2ws[frame_indices, :3, 3]
    target_positions = c2ws[target_indices, :3, 3]
    position_dists = np.linalg.norm(
        frame_positions[:, None, :] - target_positions[None, :, :],
        axis=-1,
    )
    nonzero = position_dists[position_dists > 1e-8]
    position_scale = float(np.median(nonzero)) if nonzero.size else 1.0
    position_scale = max(position_scale, 1e-6)
    position_dists = position_dists / position_scale

    rotation_dists = np.zeros_like(position_dists)
    for row, frame_idx in enumerate(frame_indices):
        rotation_a = c2ws[frame_idx, :3, :3]
        for col, target_idx in enumerate(target_indices):
            rotation_b = c2ws[target_idx, :3, :3]
            rotation_dists[row, col] = rotation_distance(rotation_a, rotation_b)

    return position_dists + rotation_weight * rotation_dists


def normalize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value < 1e-8:
        return np.ones_like(values)
    return (values - min_value) / (max_value - min_value)


def compute_rarity_irreplaceability_scores(
    c2ws,
    memory_frame_indices,
    memory_buffer,
    future_frame_indices=None,
    pinned_frames=None,
    rarity_neighbors=3,
    affinity_temperature=0.75,
):
    memory_frame_indices = list(memory_frame_indices)
    pinned_frames = set(pinned_frames or [])
    if not memory_frame_indices:
        return {}

    pairwise = pose_distances(c2ws, memory_frame_indices, memory_frame_indices)
    np.fill_diagonal(pairwise, np.inf)
    finite_pairwise = np.where(np.isfinite(pairwise), pairwise, np.nan)

    rarity = np.ones(len(memory_frame_indices), dtype=np.float64)
    if len(memory_frame_indices) > 1:
        neighbor_count = min(rarity_neighbors, len(memory_frame_indices) - 1)
        nearest = np.partition(finite_pairwise, neighbor_count - 1, axis=1)[:, :neighbor_count]
        rarity = normalize(np.nanmean(nearest, axis=1))

    irreplaceability = np.zeros(len(memory_frame_indices), dtype=np.float64)
    future_frame_indices = list(future_frame_indices or [])
    if future_frame_indices:
        future_dists = pose_distances(c2ws, memory_frame_indices, future_frame_indices)
        affinities = np.exp(-future_dists / max(affinity_temperature, 1e-6))
        affinity_sums = np.maximum(np.sum(affinities, axis=0, keepdims=True), 1e-12)
        soft_credit = affinities / affinity_sums
        irreplaceability = np.mean(soft_credit, axis=1)
        irreplaceability = normalize(irreplaceability)

    observed_use = np.array(
        [
            math.log1p(memory_buffer.selected_count(frame_idx))
            * (0.05 + memory_buffer.mean_selection_overlap(frame_idx))
            for frame_idx in memory_frame_indices
        ],
        dtype=np.float64,
    )
    observed_use = normalize(observed_use)

    scores = {}
    for index, frame_idx in enumerate(memory_frame_indices):
        score = (0.25 + rarity[index]) * (
            0.5 + irreplaceability[index] + 0.25 * observed_use[index]
        )
        if frame_idx in pinned_frames:
            score = float("inf")
        scores[frame_idx] = float(score)
    return scores
