import math
from collections import OrderedDict, defaultdict

import numpy as np


SUPPORTED_MEMORY_POLICIES = (
    "unbounded",
    "fifo",
    "rarity_irreplaceability",
    "slam_covisibility",
    "slam_max_coverage",
    "facility_coreset",
    "kcenter_coreset",
    "trajectory_coverage",
    "density_balanced_view_coverage",
    "future_view_coverage",
    "mce",
    "h2o_heavy_hitter",
    "surprise_forcing",
    "slam_ri_blend",
)
BUDGETED_MEMORY_POLICIES = (
    "fifo",
    "rarity_irreplaceability",
    "slam_covisibility",
    "slam_max_coverage",
    "facility_coreset",
    "kcenter_coreset",
    "trajectory_coverage",
    "density_balanced_view_coverage",
    "future_view_coverage",
    "mce",
    "h2o_heavy_hitter",
    "surprise_forcing",
    "slam_ri_blend",
)


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

    def remove(self, frame_idx):
        existed = frame_idx in self._frames
        self._frames.pop(frame_idx, None)
        self._stats.pop(frame_idx, None)
        return existed

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

    def stats_snapshot(self):
        return {frame_idx: dict(stats) for frame_idx, stats in self._stats.items()}

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


def camera_trajectory_similarity(
    c2ws,
    query_frame_indices,
    memory_frame_indices,
    fov_half_h=45.0,
    fov_half_v=30.0,
    radius=50.0,
):
    """Approximate direct camera-FOV overlap for trajectory coverage.

    The score is one for identical poses and falls to zero when either camera
    centers are more than two FOV radii apart or their viewing cones no longer
    overlap. Unlike ``pose_distances``, its physical scale does not change with
    the current candidate set.
    """
    query_frame_indices = list(query_frame_indices)
    memory_frame_indices = list(memory_frame_indices)
    if not query_frame_indices or not memory_frame_indices:
        return np.zeros(
            (len(query_frame_indices), len(memory_frame_indices)),
            dtype=np.float64,
        )
    if radius <= 0:
        raise ValueError("trajectory coverage radius must be positive")
    if fov_half_h <= 0 or fov_half_v <= 0:
        raise ValueError("trajectory coverage FOV half angles must be positive")

    query_poses = np.asarray(c2ws[query_frame_indices], dtype=np.float64)
    memory_poses = np.asarray(c2ws[memory_frame_indices], dtype=np.float64)

    query_positions = query_poses[:, :3, 3]
    memory_positions = memory_poses[:, :3, 3]
    position_distance = np.linalg.norm(
        query_positions[:, None, :] - memory_positions[None, :, :],
        axis=-1,
    )
    position_similarity = np.clip(
        1.0 - position_distance / (2.0 * float(radius)),
        0.0,
        1.0,
    )

    # MemCam's UE camera convention uses the first rotation column as forward.
    query_forward = query_poses[:, :3, 0]
    memory_forward = memory_poses[:, :3, 0]
    query_forward /= np.maximum(np.linalg.norm(query_forward, axis=1, keepdims=True), 1e-12)
    memory_forward /= np.maximum(np.linalg.norm(memory_forward, axis=1, keepdims=True), 1e-12)

    query_yaw = np.arctan2(query_forward[:, 1], query_forward[:, 0])
    memory_yaw = np.arctan2(memory_forward[:, 1], memory_forward[:, 0])
    yaw_distance = np.abs(query_yaw[:, None] - memory_yaw[None, :])
    yaw_distance = np.minimum(yaw_distance, 2.0 * np.pi - yaw_distance)

    query_pitch = np.arctan2(
        query_forward[:, 2],
        np.linalg.norm(query_forward[:, :2], axis=1),
    )
    memory_pitch = np.arctan2(
        memory_forward[:, 2],
        np.linalg.norm(memory_forward[:, :2], axis=1),
    )
    pitch_distance = np.abs(query_pitch[:, None] - memory_pitch[None, :])

    horizontal_similarity = np.clip(
        1.0 - yaw_distance / (2.0 * np.deg2rad(float(fov_half_h))),
        0.0,
        1.0,
    )
    vertical_similarity = np.clip(
        1.0 - pitch_distance / (2.0 * np.deg2rad(float(fov_half_v))),
        0.0,
        1.0,
    )
    return position_similarity * horizontal_similarity * vertical_similarity


def normalize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value < 1e-8:
        return np.ones_like(values)
    return (values - min_value) / (max_value - min_value)


def connected_components_from_threshold(pairwise_distances, threshold):
    num_items = pairwise_distances.shape[0]
    visited = np.zeros(num_items, dtype=bool)
    cluster_ids = np.full(num_items, -1, dtype=np.int64)
    clusters = []

    for start in range(num_items):
        if visited[start]:
            continue

        cluster_id = len(clusters)
        stack = [start]
        visited[start] = True
        members = []

        while stack:
            item = stack.pop()
            members.append(item)
            neighbors = np.flatnonzero(pairwise_distances[item] <= threshold)
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))

        for member in members:
            cluster_ids[member] = cluster_id
        clusters.append(members)

    return cluster_ids, clusters


def estimate_cluster_threshold(pairwise_distances, rarity_neighbors):
    """Median distance to each point's rarity_neighbors-th nearest neighbor.

    BUG HISTORY: this previously always used the 1st-nearest-neighbor
    distance (np.partition(..., 0)[:, 0]) regardless of rarity_neighbors,
    making the parameter dead code for every caller -- including
    compute_rarity_irreplaceability_scores (RI) and _historical_query_medoids
    (MCE's Q_hist), neither of which override cluster_distance_threshold at
    their real call sites in wan_video_memcam.py. Found via an MCE offline
    sweep showing zero variation in clustering across rarity_neighbors in
    {1,3,8,15}, confirmed with a controlled synthetic test (6 well-separated
    10-item clusters still split into 37 near-singleton clusters regardless
    of rarity_neighbors). Fixed to actually use the k-th nearest neighbor;
    rarity_neighbors=1 reproduces the old (buggy-but-now-intentional)
    behavior exactly, so this is a strict generalization, not a behavior
    change at the historical default call sites that never passed a
    non-default value anyway.
    """
    finite = pairwise_distances[np.isfinite(pairwise_distances)]
    if finite.size == 0:
        return 0.0

    # Use the median k-th-nearest-neighbor distance as the mode scale. Larger
    # k coarsens the threshold (merges more into one cluster); k=1 (default)
    # matches the original single-nearest-neighbor behavior. Larger k-neighbor
    # thresholds can merge a whole camera path into one chain-shaped component.
    # Each row has exactly one guaranteed-inf self-distance entry (diagonal),
    # which sorts to the last position after partitioning -- so the valid
    # non-self neighbor range is num_points - 2, not num_points - 1. Getting
    # this off-by-one wrong lets the index land on the self-inf entry for
    # small candidate pools, which is exactly what broke the paper's own
    # duplicate-view counterexample test at 3 candidates until this was caught.
    num_points = pairwise_distances.shape[0]
    neighbor_index = max(0, min(int(rarity_neighbors) - 1, num_points - 2))
    nearest = np.partition(pairwise_distances, neighbor_index, axis=1)[:, neighbor_index]
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size:
        return float(np.median(nearest))
    return float(np.median(finite))


def cosine_distances(features):
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-12)
    similarities = np.clip(features @ features.T, -1.0, 1.0)
    return 1.0 - similarities


def pairwise_mean_abs_distances(features):
    features = np.asarray(features, dtype=np.float32)
    distances = np.zeros((features.shape[0], features.shape[0]), dtype=np.float64)
    for index in range(features.shape[0]):
        distances[index] = np.mean(np.abs(features[index][None, :] - features), axis=1)
    return distances


def rgb_features_from_pil_images(images, image_size=64):
    features = []
    for image in images:
        image = image.convert("RGB").resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        features.append(array.reshape(-1))
    return np.stack(features, axis=0) if features else np.zeros((0, image_size * image_size * 3))


def image_quality_scores_from_pil_images(images):
    """Return simple sharpness/contrast quality scores in [0, 1]."""
    scores = []
    for image in images:
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        if gray.size == 0:
            scores.append(0.0)
            continue

        grad_y, grad_x = np.gradient(gray)
        sharpness = float(np.mean(grad_x * grad_x + grad_y * grad_y))
        contrast = float(np.std(gray))
        score = (sharpness / (sharpness + 0.002)) * (contrast / (contrast + 0.05))
        scores.append(float(np.clip(score, 0.0, 1.0)))
    return np.asarray(scores, dtype=np.float32)


class VisualMemoryFeatureExtractor:
    def __init__(
        self,
        dino_model_name="facebook/dinov2-base",
        device="cuda",
        batch_size=16,
        rgb_image_size=64,
    ):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.rgb_image_size = int(rgb_image_size)
        self.processor = AutoImageProcessor.from_pretrained(dino_model_name)
        self.model = AutoModel.from_pretrained(dino_model_name).eval().to(self.device)

    def encode_pil_images(self, images):
        dino_features = []
        with self.torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
                features = getattr(outputs, "pooler_output", None)
                if features is None:
                    features = outputs.last_hidden_state[:, 0]
                features = self.torch.nn.functional.normalize(features.float(), dim=-1)
                dino_features.append(features.detach().cpu().numpy())

        if dino_features:
            dino_features = np.concatenate(dino_features, axis=0)
        else:
            dino_features = np.zeros((0, 0), dtype=np.float32)

        rgb_features = rgb_features_from_pil_images(
            images,
            image_size=self.rgb_image_size,
        )
        return dino_features, rgb_features


def surprise_forcing_score(candidate_descriptor, bank_descriptors, alpha=0.7):
    """Equation 6 from Surprise Forcing for normalized frame descriptors."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Surprise Forcing alpha must be in [0, 1]")

    candidate = np.asarray(candidate_descriptor, dtype=np.float64).reshape(-1)
    if candidate.size == 0 or not np.all(np.isfinite(candidate)):
        raise ValueError("Candidate descriptor must be finite and non-empty")
    candidate /= max(float(np.linalg.norm(candidate)), 1e-12)

    descriptors = [
        np.asarray(descriptor, dtype=np.float64).reshape(-1)
        for descriptor in bank_descriptors
    ]
    if not descriptors:
        return {
            "surprise": 1.0,
            "prediction_surprise": 1.0,
            "novelty_surprise": 1.0,
            "mean_similarity": -1.0,
            "max_similarity": -1.0,
        }

    matrix = np.stack(descriptors)
    if matrix.shape[1] != candidate.shape[0] or not np.all(np.isfinite(matrix)):
        raise ValueError("Bank descriptors must be finite and match candidate size")
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    similarities = np.clip(matrix @ candidate, -1.0, 1.0)
    mean_similarity = float(np.mean(similarities))
    max_similarity = float(np.max(similarities))
    prediction_surprise = 0.5 * (1.0 - mean_similarity)
    novelty_surprise = 0.5 * (1.0 - max_similarity)
    surprise = alpha * prediction_surprise + (1.0 - alpha) * novelty_surprise
    return {
        "surprise": float(surprise),
        "prediction_surprise": float(prediction_surprise),
        "novelty_surprise": float(novelty_surprise),
        "mean_similarity": mean_similarity,
        "max_similarity": max_similarity,
    }


class SurpriseForcingMemoryController:
    """Causal Surprise-Gated Memory Bank adapted to MemCam frame payloads.

    The paper specifies that surprise, usage, and age are normalized to [0, 1]
    before priority scoring, but not the normalization operator. Surprise is
    already bounded by Equation 6; usage and age are divided by their current
    maxima. EMA state starts at mean 0 and variance 1 because its initialization
    is not disclosed. The paper's unspecified duplicate and neighboring-chunk
    filters are deliberately omitted.
    """

    def __init__(
        self,
        capacity,
        alpha=0.7,
        ema_momentum=0.95,
        controller_step=0.1,
        target_admission_ratio=0.3,
        initial_threshold=0.002,
        surprise_weight=1.8,
        usage_weight=1.0,
        age_weight=0.4,
        warmup_sections=3,
        epsilon=1e-6,
    ):
        if capacity < 1:
            raise ValueError("Surprise Forcing capacity must be positive")
        if not 0.0 <= ema_momentum < 1.0:
            raise ValueError("EMA momentum must be in [0, 1)")
        if not 0.0 <= target_admission_ratio <= 1.0:
            raise ValueError("Target admission ratio must be in [0, 1]")
        if controller_step < 0 or epsilon <= 0:
            raise ValueError("Controller step must be non-negative and epsilon positive")
        if warmup_sections < 0:
            raise ValueError("Warmup sections must be non-negative")

        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.ema_momentum = float(ema_momentum)
        self.controller_step = float(controller_step)
        self.target_admission_ratio = float(target_admission_ratio)
        self.initial_threshold = float(initial_threshold)
        self.threshold = float(initial_threshold)
        self.surprise_weight = float(surprise_weight)
        self.usage_weight = float(usage_weight)
        self.age_weight = float(age_weight)
        self.warmup_sections = int(warmup_sections)
        self.epsilon = float(epsilon)

        self.ema_mean = 0.0
        self.ema_variance = 1.0
        self.evaluated_count = 0
        self.gate_pass_count = 0
        self.commit_count = 0
        self._warmup_priorities_reset = False
        self._entries = OrderedDict()

    def frames(self):
        return list(self._entries)

    def descriptor(self, frame_idx):
        entry = self._entries.get(frame_idx)
        return None if entry is None else entry["descriptor"]

    def maybe_finish_warmup(self, section_idx):
        if self._warmup_priorities_reset or section_idx < self.warmup_sections:
            return False
        reset_value = float(np.clip(self.ema_mean, 0.0, 1.0))
        for entry in self._entries.values():
            entry["write_surprise"] = reset_value
        self._warmup_priorities_reset = True
        return True

    @staticmethod
    def _divide_by_max(values):
        values = np.asarray(values, dtype=np.float64)
        maximum = float(np.max(values)) if values.size else 0.0
        if maximum <= 1e-12:
            return np.zeros_like(values)
        return values / maximum

    def _priority_rows(self, current_frame, pending=None):
        frame_indices = list(self._entries)
        surprises = [self._entries[idx]["write_surprise"] for idx in frame_indices]
        usages = [self._entries[idx]["usage"] for idx in frame_indices]
        ages = [max(int(current_frame) - int(idx), 0) for idx in frame_indices]
        if pending is not None:
            frame_indices.append(int(pending["frame_idx"]))
            surprises.append(float(pending["write_surprise"]))
            usages.append(0.0)
            ages.append(0.0)

        surprise_norm = np.clip(np.asarray(surprises, dtype=np.float64), 0.0, 1.0)
        usage_norm = self._divide_by_max(usages)
        age_norm = self._divide_by_max(ages)
        priorities = (
            self.surprise_weight * surprise_norm
            + self.usage_weight * usage_norm
            - self.age_weight * age_norm
        )
        return {
            frame_idx: {
                "priority": float(priorities[index]),
                "surprise_normalized": float(surprise_norm[index]),
                "usage_normalized": float(usage_norm[index]),
                "age_normalized": float(age_norm[index]),
                "usage": int(usages[index]),
                "age": int(ages[index]),
            }
            for index, frame_idx in enumerate(frame_indices)
        }

    def consider(self, frame_idx, descriptor, section_idx, current_frame=None):
        frame_idx = int(frame_idx)
        current_frame = frame_idx if current_frame is None else int(current_frame)
        if frame_idx in self._entries:
            raise ValueError(f"Frame {frame_idx} is already in the Surprise Forcing bank")
        self.maybe_finish_warmup(section_idx)

        score = surprise_forcing_score(
            descriptor,
            [entry["descriptor"] for entry in self._entries.values()],
            alpha=self.alpha,
        )
        sigma_before = math.sqrt(max(self.ema_variance, self.epsilon))
        mean_before = float(self.ema_mean)
        variance_before = float(self.ema_variance)
        z_score = (score["surprise"] - mean_before) / sigma_before
        underfilled = len(self._entries) < self.capacity
        active_threshold = (
            min(self.threshold, self.initial_threshold)
            if underfilled
            else self.threshold
        )
        warmup = int(section_idx) < self.warmup_sections
        gate_pass = bool(warmup or z_score >= active_threshold)

        new_mean = (
            self.ema_momentum * self.ema_mean
            + (1.0 - self.ema_momentum) * score["surprise"]
        )
        new_variance = (
            self.ema_momentum * self.ema_variance
            + (1.0 - self.ema_momentum) * (score["surprise"] - new_mean) ** 2
        )
        self.ema_mean = float(new_mean)
        self.ema_variance = float(max(new_variance, self.epsilon))
        threshold_before = float(self.threshold)
        if not warmup:
            self.threshold = float(
                np.clip(
                    self.threshold
                    + self.controller_step
                    * (float(gate_pass) - self.target_admission_ratio),
                    -3.0,
                    3.0,
                )
            )
        self.evaluated_count += 1
        self.gate_pass_count += int(gate_pass)

        decision = {
            "frame_idx": frame_idx,
            "section_idx": int(section_idx),
            **score,
            "z_score": float(z_score),
            "ema_mean_before": mean_before,
            "ema_variance_before": variance_before,
            "ema_mean_after": float(self.ema_mean),
            "ema_variance_after": float(self.ema_variance),
            "sigma_before": float(sigma_before),
            "threshold_before": threshold_before,
            "active_threshold": float(active_threshold),
            "threshold_after": float(self.threshold),
            "target_admission_ratio": self.target_admission_ratio,
            "warmup": bool(warmup),
            "gate_pass": gate_pass,
            "committed": False,
            "evicted_frame": None,
            "rejection_reason": None,
            "candidate_priority": None,
            "minimum_bank_priority": None,
        }
        if not gate_pass:
            decision["rejection_reason"] = "surprise_gate"
            return decision

        descriptor = np.asarray(descriptor, dtype=np.float32).reshape(-1)
        descriptor /= max(float(np.linalg.norm(descriptor)), 1e-12)
        pending = {
            "frame_idx": frame_idx,
            "write_surprise": score["surprise"],
        }
        priority_rows = self._priority_rows(current_frame, pending=pending)
        candidate_priority = priority_rows[frame_idx]["priority"]
        decision["candidate_priority"] = candidate_priority

        evicted_frame = None
        if len(self._entries) >= self.capacity:
            bank_priorities = {
                idx: priority_rows[idx]["priority"] for idx in self._entries
            }
            evicted_frame = min(
                bank_priorities,
                key=lambda idx: (bank_priorities[idx], self._entries[idx]["write_order"]),
            )
            minimum_priority = bank_priorities[evicted_frame]
            decision["minimum_bank_priority"] = float(minimum_priority)
            if candidate_priority <= minimum_priority:
                decision["rejection_reason"] = "priority"
                decision["evicted_frame"] = None
                return decision
            self._entries.pop(evicted_frame)

        self._entries[frame_idx] = {
            "descriptor": descriptor,
            "write_surprise": float(score["surprise"]),
            "usage": 0,
            "write_order": self.evaluated_count,
        }
        self.commit_count += 1
        decision["committed"] = True
        decision["evicted_frame"] = evicted_frame
        return decision

    def record_usage(self, frame_indices):
        for frame_idx in set(int(value) for value in frame_indices):
            if frame_idx in self._entries:
                self._entries[frame_idx]["usage"] += 1

    def route(
        self,
        query_descriptor,
        candidate_frames=None,
        top_k=3,
        record_usage=True,
    ):
        if top_k < 1:
            raise ValueError("Surprise Forcing top_k must be positive")
        candidates = (
            self.frames() if candidate_frames is None else list(candidate_frames)
        )
        candidates = [frame_idx for frame_idx in candidates if frame_idx in self._entries]
        if not candidates:
            return [], {}

        query = np.asarray(query_descriptor, dtype=np.float64).reshape(-1)
        if query.size == 0 or not np.all(np.isfinite(query)):
            raise ValueError("Query descriptor must be finite and non-empty")
        query /= max(float(np.linalg.norm(query)), 1e-12)
        similarities = {}
        for frame_idx in candidates:
            descriptor = self._entries[frame_idx]["descriptor"]
            if descriptor.shape != query.shape:
                raise ValueError(
                    "Query descriptor must match Surprise Forcing bank descriptors"
                )
            similarities[frame_idx] = float(
                np.clip(np.dot(query, descriptor), -1.0, 1.0)
            )
        routed = sorted(
            candidates,
            key=lambda idx: (-similarities[idx], -idx),
        )[: min(int(top_k), len(candidates))]
        if record_usage:
            self.record_usage(routed)
        return routed, similarities

    def state_snapshot(self, current_frame):
        priority_rows = self._priority_rows(current_frame)
        return {
            frame_idx: {
                "write_surprise": float(entry["write_surprise"]),
                "usage": int(entry["usage"]),
                **priority_rows[frame_idx],
            }
            for frame_idx, entry in self._entries.items()
        }


def compute_rarity_irreplaceability_scores(
    memory_frame_indices,
    pinned_frames=None,
    rarity_neighbors=3,
    cluster_distance_threshold=None,
    return_details=False,
    dino_features=None,
    rgb_features=None,
):
    memory_frame_indices = list(memory_frame_indices)
    pinned_frames = set(pinned_frames or [])
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}

    if dino_features is None or rgb_features is None:
        raise ValueError(
            "rarity_irreplaceability now requires DINO features for rarity "
            "and RGB features for irreplaceability."
        )

    missing_features = [
        frame_idx
        for frame_idx in memory_frame_indices
        if frame_idx not in dino_features or frame_idx not in rgb_features
    ]
    if missing_features:
        raise ValueError(f"Missing visual memory features for frames: {missing_features[:10]}")

    dino_matrix = np.stack([dino_features[frame_idx] for frame_idx in memory_frame_indices])
    rgb_matrix = np.stack([rgb_features[frame_idx] for frame_idx in memory_frame_indices])

    dino_pairwise = cosine_distances(dino_matrix)
    np.fill_diagonal(dino_pairwise, np.inf)

    if len(memory_frame_indices) == 1:
        cluster_ids = np.zeros(1, dtype=np.int64)
        cluster_sizes = np.ones(1, dtype=np.float64)
        threshold = 0.0
    else:
        threshold = (
            float(cluster_distance_threshold)
            if cluster_distance_threshold is not None
            else estimate_cluster_threshold(dino_pairwise, rarity_neighbors)
        )
        cluster_pairwise = dino_pairwise.copy()
        np.fill_diagonal(cluster_pairwise, 0.0)
        cluster_ids, clusters = connected_components_from_threshold(
            cluster_pairwise,
            threshold=threshold,
        )
        cluster_sizes = np.array([len(clusters[cluster_id]) for cluster_id in cluster_ids])

    memory_count = float(len(memory_frame_indices))
    rarity = np.log((memory_count + 1.0) / np.maximum(cluster_sizes, 1.0))

    rgb_pairwise = pairwise_mean_abs_distances(rgb_matrix)
    np.fill_diagonal(rgb_pairwise, np.inf)
    if len(memory_frame_indices) == 1:
        nearest_rgb_distances = np.ones(1, dtype=np.float64)
        nearest_rgb_indices = np.full(1, -1, dtype=np.int64)
    else:
        nearest_rgb_indices = np.argmin(rgb_pairwise, axis=1)
        nearest_rgb_distances = rgb_pairwise[np.arange(len(memory_frame_indices)), nearest_rgb_indices]
    irreplaceability = nearest_rgb_distances

    scores = {}
    details = {}
    for index, frame_idx in enumerate(memory_frame_indices):
        score = rarity[index] * irreplaceability[index]
        if frame_idx in pinned_frames:
            score = float("inf")
        scores[frame_idx] = float(score)
        details[frame_idx] = {
            "score": float(score),
            "rarity": float(rarity[index]),
            "irreplaceability": float(irreplaceability[index]),
            "cluster_id": int(cluster_ids[index]),
            "cluster_size": int(cluster_sizes[index]),
            "cluster_threshold": float(threshold),
            "rgb_nearest_frame": (
                None
                if nearest_rgb_indices[index] < 0
                else int(memory_frame_indices[int(nearest_rgb_indices[index])])
            ),
            "rgb_nearest_distance": float(nearest_rgb_distances[index]),
        }
    return (scores, details) if return_details else scores


def _feature_cosine_similarity(memory_frame_indices, features):
    missing_features = [frame_idx for frame_idx in memory_frame_indices if frame_idx not in features]
    if missing_features:
        raise ValueError(f"Missing visual memory features for frames: {missing_features[:10]}")

    feature_matrix = np.asarray(
        np.stack([features[frame_idx] for frame_idx in memory_frame_indices]),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(feature_matrix)):
        raise ValueError("Visual memory features must be finite")
    norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
    feature_matrix = feature_matrix / np.maximum(norms, 1e-12)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        similarities = feature_matrix @ feature_matrix.T
    if not np.all(np.isfinite(similarities)):
        raise ValueError("Visual cosine similarities must be finite")
    return np.clip(similarities, -1.0, 1.0)


def _slam_covisibility_affinity(
    memory_frame_indices,
    c2ws,
    dino_features=None,
    rgb_features=None,
    visual_weight=0.35,
    geometry_weight=0.65,
    self_similarity=0.0,
):
    """Build the pose/appearance affinity shared by the SLAM policies."""
    memory_frame_indices = list(memory_frame_indices)
    if not memory_frame_indices:
        return np.zeros((0, 0), dtype=np.float64)

    pose_distance = pose_distances(
        c2ws,
        memory_frame_indices,
        memory_frame_indices,
    )
    geom_similarity = np.exp(-pose_distance)

    components = [(geometry_weight, geom_similarity)]
    if dino_features is not None:
        visual_similarity = _feature_cosine_similarity(
            memory_frame_indices,
            dino_features,
        )
        components.append((visual_weight, np.maximum(visual_similarity, 0.0)))
    elif rgb_features is not None:
        rgb_distance = pairwise_mean_abs_distances(
            np.stack(
                [rgb_features[frame_idx] for frame_idx in memory_frame_indices]
            )
        )
        components.append((visual_weight, np.exp(-4.0 * rgb_distance)))

    total_weight = sum(weight for weight, _ in components)
    affinity = sum(weight * matrix for weight, matrix in components) / max(
        total_weight,
        1e-12,
    )
    affinity = np.clip(affinity, 0.0, 1.0)
    np.fill_diagonal(affinity, float(self_similarity))
    return affinity


def compute_slam_covisibility_scores(
    memory_frame_indices,
    c2ws,
    pinned_frames=None,
    dino_features=None,
    rgb_features=None,
    n_other_observers=3,
    covisibility_threshold=0.65,
    visual_weight=0.35,
    geometry_weight=0.65,
    return_details=False,
):
    """Score frames by SLAM-style marginal evidence contribution.

    Higher scores mean the frame is more worth keeping. A low score means the
    frame is already covered by several visually/geometrically similar memories.
    """
    memory_frame_indices = list(memory_frame_indices)
    pinned_frames = set(pinned_frames or [])
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}

    covisibility = _slam_covisibility_affinity(
        memory_frame_indices=memory_frame_indices,
        c2ws=c2ws,
        dino_features=dino_features,
        rgb_features=rgb_features,
        visual_weight=visual_weight,
        geometry_weight=geometry_weight,
        self_similarity=0.0,
    )

    scores = {}
    details = {}
    for row, frame_idx in enumerate(memory_frame_indices):
        row_values = covisibility[row]
        observer_indices = np.flatnonzero(row_values >= covisibility_threshold)
        covisible_observers = int(observer_indices.size)
        redundancy_ratio = min(covisible_observers / max(float(n_other_observers), 1.0), 1.0)

        if row_values.size:
            nearest_index = int(np.argmax(row_values))
            nearest_frame = int(memory_frame_indices[nearest_index])
            max_covisibility = float(row_values[nearest_index])
        else:
            nearest_frame = None
            max_covisibility = 0.0

        marginal_contribution = 1.0 / (covisible_observers + 1.0)
        unique_bonus = 1.0 - max_covisibility
        score = (1.0 - redundancy_ratio) + 0.5 * marginal_contribution + 0.25 * unique_bonus
        if frame_idx in pinned_frames:
            score = float("inf")

        scores[frame_idx] = float(score)
        details[frame_idx] = {
            "score": float(score),
            "redundancy_ratio": float(redundancy_ratio),
            "covisible_observers": covisible_observers,
            "max_covisibility": float(max_covisibility),
            "nearest_covisible_frame": nearest_frame,
            "marginal_contribution": float(marginal_contribution),
            "unique_bonus": float(unique_bonus),
            "covisibility_threshold": float(covisibility_threshold),
            "n_other_observers": int(n_other_observers),
        }

    return (scores, details) if return_details else scores


def greedy_max_coverage_selection(
    affinity,
    budget,
    demand_weights=None,
    forced_candidate_indices=None,
):
    """Maximize weighted top-1 coverage for any fixed affinity matrix.

    Rows are causal demand queries and columns are candidate memory items. The
    optimizer is representation-independent: an adapter only needs to provide
    affinities in ``[0, 1]`` and optional non-negative query weights.
    """
    affinity = np.asarray(affinity, dtype=np.float64)
    if affinity.ndim != 2:
        raise ValueError("coverage affinity must be a two-dimensional matrix")
    if not np.all(np.isfinite(affinity)):
        raise ValueError("coverage affinity must be finite")
    if np.any(affinity < 0.0) or np.any(affinity > 1.0):
        raise ValueError("coverage affinity values must be in [0, 1]")

    num_queries, num_candidates = affinity.shape
    if num_queries < 1 or num_candidates < 1:
        raise ValueError("coverage affinity must contain queries and candidates")
    if budget is None or budget <= 0:
        raise ValueError("max coverage requires a positive budget")

    if demand_weights is None:
        weights = np.ones(num_queries, dtype=np.float64)
    else:
        weights = np.asarray(demand_weights, dtype=np.float64)
        if weights.shape != (num_queries,):
            raise ValueError(
                "coverage demand weights must match the affinity query count"
            )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("coverage demand weights must be finite and non-negative")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("coverage demand weights must have positive total mass")
    weights = weights / weight_sum

    forced_cols = list(dict.fromkeys(forced_candidate_indices or []))
    invalid_forced = [
        col for col in forced_cols if col < 0 or col >= num_candidates
    ]
    if invalid_forced:
        raise ValueError(
            f"forced coverage candidate indices are invalid: {invalid_forced[:10]}"
        )
    if len(forced_cols) > budget:
        raise ValueError("max coverage has more forced candidates than its budget")

    selected_cols = list(forced_cols)
    selected_set = set(forced_cols)
    marginal_gains = {col: float("inf") for col in forced_cols}
    covered = (
        np.max(affinity[:, selected_cols], axis=1)
        if selected_cols
        else np.zeros(num_queries, dtype=np.float64)
    )

    selected_limit = min(int(budget), num_candidates)
    while len(selected_cols) < selected_limit:
        gains = np.sum(
            weights[:, None]
            * np.maximum(0.0, affinity - covered[:, None]),
            axis=0,
        )
        if selected_set:
            gains[list(selected_set)] = -np.inf
        best_col = int(np.argmax(gains))
        if not np.isfinite(gains[best_col]):
            break

        selected_cols.append(best_col)
        selected_set.add(best_col)
        marginal_gains[best_col] = float(gains[best_col])
        covered = np.maximum(covered, affinity[:, best_col])

    removal_losses = {}
    for col in selected_cols:
        other_cols = [other for other in selected_cols if other != col]
        covered_without_col = (
            np.max(affinity[:, other_cols], axis=1)
            if other_cols
            else np.zeros(num_queries, dtype=np.float64)
        )
        removal_losses[col] = float(
            np.sum(weights * (covered - covered_without_col))
        )

    candidate_gains = np.sum(
        weights[:, None]
        * np.maximum(0.0, affinity - covered[:, None]),
        axis=0,
    )
    if selected_set:
        candidate_gains[list(selected_set)] = 0.0

    return {
        "selected_cols": selected_cols,
        "marginal_gains": marginal_gains,
        "removal_losses": removal_losses,
        "candidate_gains": candidate_gains,
        "covered": covered,
        "coverage_value": float(np.sum(weights * covered)),
        "demand_weights": weights,
    }


def compute_slam_max_coverage_scores(
    memory_frame_indices,
    c2ws,
    budget,
    forced_keep_frames=None,
    dino_features=None,
    rgb_features=None,
    visual_weight=0.35,
    geometry_weight=0.65,
    return_details=False,
):
    """Select the subset with maximum top-1 coverage under SLAM affinity.

    Queries and candidates are the current causal set ``M_(t-1) union N_t``.
    Every query has uniform demand. Off-diagonal affinities exactly match the
    SLAM covisibility policy; the diagonal is one because a retained memory
    fully covers itself. Greedy maximizes the monotone submodular facility
    objective ``mean_q max_{m in M} K(q, m)``.
    """
    memory_frame_indices = list(memory_frame_indices)
    forced_keep_frames = set(forced_keep_frames or [])
    if budget is None:
        raise ValueError("slam_max_coverage requires an explicit memory budget")
    if budget <= 0:
        raise ValueError("slam_max_coverage budget must be positive")
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}
    if len(set(memory_frame_indices)) != len(memory_frame_indices):
        raise ValueError("slam_max_coverage candidates must be unique")

    candidate_set = set(memory_frame_indices)
    unknown_forced = forced_keep_frames - candidate_set
    if unknown_forced:
        raise ValueError(
            "slam_max_coverage forced-keep frames are not candidates: "
            f"{sorted(unknown_forced)[:10]}"
        )
    if len(forced_keep_frames) > budget:
        raise ValueError("slam_max_coverage has more forced frames than its budget")

    affinity = _slam_covisibility_affinity(
        memory_frame_indices=memory_frame_indices,
        c2ws=c2ws,
        dino_features=dino_features,
        rgb_features=rgb_features,
        visual_weight=visual_weight,
        geometry_weight=geometry_weight,
        self_similarity=1.0,
    )
    num_candidates = len(memory_frame_indices)
    frame_to_col = {
        frame_idx: col for col, frame_idx in enumerate(memory_frame_indices)
    }
    forced_cols = [
        frame_to_col[frame_idx]
        for frame_idx in memory_frame_indices
        if frame_idx in forced_keep_frames
    ]
    selection = greedy_max_coverage_selection(
        affinity=affinity,
        budget=budget,
        forced_candidate_indices=forced_cols,
    )
    selected_cols = selection["selected_cols"]
    marginal_gains = selection["marginal_gains"]
    covered = selection["covered"]

    selected_frames = [memory_frame_indices[col] for col in selected_cols]
    selected_frame_set = set(selected_frames)
    selected_ranks = {col: rank for rank, col in enumerate(selected_cols)}
    coverage_value = selection["coverage_value"]
    removal_losses = selection["removal_losses"]
    candidate_gains = selection["candidate_gains"]

    visual_source = (
        "dino"
        if dino_features is not None
        else "rgb" if rgb_features is not None else "none"
    )
    scores = {}
    details = {}
    for col, frame_idx in enumerate(memory_frame_indices):
        selected = frame_idx in selected_frame_set
        forced = frame_idx in forced_keep_frames
        if forced:
            score = float("inf")
        elif selected:
            score = 1.0 + removal_losses.get(col, 0.0)
        else:
            score = -1.0

        candidate_gain = (
            0.0
            if selected
            else float(candidate_gains[col])
        )
        gain = marginal_gains.get(col)
        scores[frame_idx] = float(score)
        details[frame_idx] = {
            "score": float(score),
            "slam_max_coverage_selected": bool(selected),
            "slam_max_coverage_forced_keep": bool(forced),
            "slam_max_coverage_rank": selected_ranks.get(col),
            "slam_max_coverage_marginal_gain": (
                None if gain is None or np.isinf(gain) else float(gain)
            ),
            "slam_max_coverage_candidate_gain": candidate_gain,
            "slam_max_coverage_removal_loss": (
                float(removal_losses.get(col, 0.0)) if selected else 0.0
            ),
            "slam_max_coverage_value": coverage_value,
            "slam_max_coverage_mean": coverage_value,
            "slam_max_coverage_min": float(np.min(covered)),
            "slam_max_coverage_p10": float(np.quantile(covered, 0.10)),
            "slam_max_coverage_affinity_mean": float(
                np.mean(affinity[:, col])
            ),
            "slam_max_coverage_affinity_max": float(
                np.max(affinity[:, col])
            ),
            "slam_max_coverage_geometry_weight": float(geometry_weight),
            "slam_max_coverage_visual_weight": float(visual_weight),
            "slam_max_coverage_visual_source": visual_source,
            "slam_max_coverage_query_count": num_candidates,
        }

    return (scores, details) if return_details else scores


def _feature_cosine_similarity_cross(left_frame_indices, right_frame_indices, features):
    frame_indices = list(left_frame_indices) + list(right_frame_indices)
    missing_features = [frame_idx for frame_idx in frame_indices if frame_idx not in features]
    if missing_features:
        raise ValueError(f"Missing visual memory features for frames: {missing_features[:10]}")

    left = np.stack([features[frame_idx] for frame_idx in left_frame_indices])
    right = np.stack([features[frame_idx] for frame_idx in right_frame_indices])
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return np.clip(left @ right.T, -1.0, 1.0)


def compute_facility_coreset_scores(
    memory_frame_indices,
    archive_frame_indices,
    c2ws,
    budget,
    forced_keep_frames=None,
    dino_features=None,
    frame_quality=None,
    archive_weights=None,
    visual_weight=0.6,
    pose_weight=0.4,
    time_weight=0.0,
    similarity_sigma=0.35,
    return_details=False,
):
    """Select representative memory frames with a facility-location coreset.

    The archive defines all historical descriptors to represent. The memory set
    contains actual frames the model can retrieve. Higher returned scores mean
    "keep"; unselected frames get low scores and are evicted by the buffer.
    """
    memory_frame_indices = list(memory_frame_indices)
    archive_frame_indices = list(archive_frame_indices)
    forced_keep_frames = set(forced_keep_frames or [])
    if budget is None:
        raise ValueError("facility_coreset requires an explicit memory budget")
    if budget <= 0:
        raise ValueError("facility_coreset budget must be positive")
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}
    if not archive_frame_indices:
        archive_frame_indices = list(memory_frame_indices)
    if dino_features is None:
        raise ValueError("facility_coreset requires DINO features")

    missing_features = [
        frame_idx
        for frame_idx in set(memory_frame_indices) | set(archive_frame_indices)
        if frame_idx not in dino_features
    ]
    if missing_features:
        raise ValueError(f"Missing coreset DINO features for frames: {missing_features[:10]}")

    if len(memory_frame_indices) <= budget:
        selected_frames = list(memory_frame_indices)
        frame_quality = frame_quality or {}
        scores = {
            frame_idx: float("inf") if frame_idx in forced_keep_frames else 1.0
            for frame_idx in memory_frame_indices
        }
        details = {
            frame_idx: {
                "score": scores[frame_idx],
                "coreset_selected": True,
                "coreset_rank": index,
                "coreset_marginal_gain": None,
                "coreset_removal_loss": None,
                "coreset_archive_size": len(archive_frame_indices),
                "coreset_facility_value": None,
                "coreset_quality": float(frame_quality.get(frame_idx, 1.0)),
            }
            for index, frame_idx in enumerate(selected_frames)
        }
        return (scores, details) if return_details else scores

    weights = (
        np.ones(len(archive_frame_indices), dtype=np.float64)
        if archive_weights is None
        else np.asarray(archive_weights, dtype=np.float64)
    )
    if weights.shape[0] != len(archive_frame_indices):
        raise ValueError("archive_weights length must match archive_frame_indices")
    weights = np.maximum(weights, 0.0)

    visual_similarity = _feature_cosine_similarity_cross(
        archive_frame_indices,
        memory_frame_indices,
        dino_features,
    )
    visual_distance = 1.0 - visual_similarity

    pose_distance = pose_distances(c2ws, archive_frame_indices, memory_frame_indices)
    pose_distance = 1.0 - np.exp(-pose_distance)

    distance = visual_weight * visual_distance + pose_weight * pose_distance
    if time_weight:
        archive_times = np.asarray(archive_frame_indices, dtype=np.float64)
        memory_times = np.asarray(memory_frame_indices, dtype=np.float64)
        time_scale = max(
            float(max(max(archive_frame_indices), max(memory_frame_indices)) + 1),
            1.0,
        )
        time_distance = np.abs(archive_times[:, None] - memory_times[None, :]) / time_scale
        distance = distance + float(time_weight) * time_distance

    total_weight = max(float(visual_weight + pose_weight + time_weight), 1e-12)
    distance = distance / total_weight
    similarity = np.exp(-distance / max(float(similarity_sigma), 1e-12))

    qualities = np.ones(len(memory_frame_indices), dtype=np.float64)
    if frame_quality is not None:
        qualities = np.asarray(
            [frame_quality.get(frame_idx, 1.0) for frame_idx in memory_frame_indices],
            dtype=np.float64,
        )
        qualities = np.clip(qualities, 0.0, 1.0)
    similarity = similarity * qualities[None, :]

    frame_to_col = {frame_idx: col for col, frame_idx in enumerate(memory_frame_indices)}
    forced_cols = [
        frame_to_col[frame_idx]
        for frame_idx in memory_frame_indices
        if frame_idx in forced_keep_frames
    ]

    selected_cols = []
    selected_set = set()
    marginal_gains = {}

    for col in forced_cols:
        if col not in selected_set:
            selected_set.add(col)
            selected_cols.append(col)
            marginal_gains[col] = float("inf")

    if selected_cols:
        covered = np.max(similarity[:, selected_cols], axis=1)
    else:
        covered = np.zeros(len(archive_frame_indices), dtype=np.float64)

    while len(selected_cols) < min(int(budget), len(memory_frame_indices)):
        best_col = None
        best_gain = -float("inf")
        for col in range(len(memory_frame_indices)):
            if col in selected_set:
                continue
            gain = float(np.sum(weights * np.maximum(0.0, similarity[:, col] - covered)))
            if gain > best_gain:
                best_gain = gain
                best_col = col

        if best_col is None:
            break

        selected_set.add(best_col)
        selected_cols.append(best_col)
        marginal_gains[best_col] = best_gain
        covered = np.maximum(covered, similarity[:, best_col])

    selected_frames = [memory_frame_indices[col] for col in selected_cols]
    selected_frame_set = set(selected_frames)
    facility_value = float(np.sum(weights * covered))

    removal_losses = {}
    for col in selected_cols:
        other_cols = [other for other in selected_cols if other != col]
        if other_cols:
            without_col = np.max(similarity[:, other_cols], axis=1)
        else:
            without_col = np.zeros(len(archive_frame_indices), dtype=np.float64)
        removal_losses[col] = float(np.sum(weights * (covered - without_col)))

    scores = {}
    details = {}
    for col, frame_idx in enumerate(memory_frame_indices):
        selected = frame_idx in selected_frame_set
        forced = frame_idx in forced_keep_frames
        if forced:
            score = float("inf")
        elif selected:
            score = 1.0 + removal_losses.get(col, 0.0)
        else:
            score = -1.0

        rank = selected_frames.index(frame_idx) if selected else None
        gain = marginal_gains.get(col)
        candidate_gain = float(np.sum(weights * np.maximum(0.0, similarity[:, col] - covered)))
        scores[frame_idx] = float(score)
        details[frame_idx] = {
            "score": float(score),
            "coreset_selected": bool(selected),
            "coreset_forced_keep": bool(forced),
            "coreset_rank": rank,
            "coreset_marginal_gain": None if gain is None or np.isinf(gain) else float(gain),
            "coreset_candidate_gain": float(candidate_gain),
            "coreset_removal_loss": float(removal_losses.get(col, 0.0)) if selected else 0.0,
            "coreset_archive_size": len(archive_frame_indices),
            "coreset_facility_value": facility_value,
            "coreset_quality": float(qualities[col]),
            "coreset_similarity_mean": float(np.mean(similarity[:, col])),
            "coreset_similarity_max": float(np.max(similarity[:, col])),
            "coreset_visual_weight": float(visual_weight),
            "coreset_pose_weight": float(pose_weight),
            "coreset_time_weight": float(time_weight),
            "coreset_similarity_sigma": float(similarity_sigma),
        }

    return (scores, details) if return_details else scores


def compute_density_balanced_view_coverage_scores(
    memory_frame_indices,
    c2ws,
    budget,
    forced_keep_frames=None,
    dino_features=None,
    rgb_features=None,
    coverage_masses=None,
    density_alpha=0.5,
    dino_weight=0.5,
    rgb_weight=0.25,
    density_epsilon=1e-3,
    fov_half_h=45.0,
    fov_half_v=30.0,
    radius=50.0,
    return_details=False,
):
    """Select a bounded online coreset by probabilistic view coverage.

    The candidate and query universe is exactly the retained bank plus the new
    frames. Each retained query carries the mass of observations it represented
    during the previous update, so no separate historical descriptor archive is
    required. Density-balanced demand protects sparse view regions, while a soft
    pose/DINO/RGB kernel estimates whether one memory can substitute for another.
    With fixed demand weights and kernel, the noisy-OR set objective is monotone
    submodular, so the budgeted subset is built by forward greedy selection.
    """
    memory_frame_indices = list(memory_frame_indices)
    forced_keep_frames = set(forced_keep_frames or [])
    if budget is None:
        raise ValueError(
            "density_balanced_view_coverage requires an explicit memory budget"
        )
    if budget <= 0:
        raise ValueError("density_balanced_view_coverage budget must be positive")
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}
    if len(set(memory_frame_indices)) != len(memory_frame_indices):
        raise ValueError("density_balanced_view_coverage candidates must be unique")
    if dino_features is None or rgb_features is None:
        raise ValueError(
            "density_balanced_view_coverage requires DINO and RGB features"
        )
    if not 0.0 <= density_alpha <= 1.0:
        raise ValueError("density_alpha must be in [0, 1]")
    if dino_weight < 0 or rgb_weight < 0:
        raise ValueError("density kernel weights must be non-negative")
    if density_epsilon <= 0:
        raise ValueError("density_epsilon must be positive")

    candidate_set = set(memory_frame_indices)
    unknown_forced = forced_keep_frames - candidate_set
    if unknown_forced:
        raise ValueError(
            "density_balanced_view_coverage forced-keep frames are not candidates: "
            f"{sorted(unknown_forced)[:10]}"
        )
    if len(forced_keep_frames) > budget:
        raise ValueError(
            "density_balanced_view_coverage has more forced frames than its budget"
        )

    missing_dino = [
        frame_idx
        for frame_idx in memory_frame_indices
        if frame_idx not in dino_features
    ]
    missing_rgb = [
        frame_idx
        for frame_idx in memory_frame_indices
        if frame_idx not in rgb_features
    ]
    if missing_dino or missing_rgb:
        raise ValueError(
            "Missing density coverage features for frames: "
            f"DINO={missing_dino[:10]}, RGB={missing_rgb[:10]}"
        )

    coverage_masses = coverage_masses or {}
    base_masses = np.asarray(
        [coverage_masses.get(frame_idx, 1.0) for frame_idx in memory_frame_indices],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(base_masses)) or np.any(base_masses < 0):
        raise ValueError("coverage masses must be finite and non-negative")
    total_mass = float(np.sum(base_masses))
    if total_mass <= 0:
        raise ValueError("coverage masses must have positive total mass")

    geometry_similarity = camera_trajectory_similarity(
        c2ws=c2ws,
        query_frame_indices=memory_frame_indices,
        memory_frame_indices=memory_frame_indices,
        fov_half_h=fov_half_h,
        fov_half_v=fov_half_v,
        radius=radius,
    )
    dino_similarity = _feature_cosine_similarity(
        memory_frame_indices,
        dino_features,
    )
    dino_distance = np.clip(1.0 - dino_similarity, 0.0, 2.0)
    rgb_matrix = np.stack(
        [rgb_features[frame_idx] for frame_idx in memory_frame_indices]
    )
    rgb_distance = pairwise_mean_abs_distances(rgb_matrix)
    if not np.all(np.isfinite(rgb_distance)):
        raise ValueError("RGB memory features must be finite")
    appearance_similarity = np.exp(
        -float(dino_weight) * dino_distance
        -float(rgb_weight) * rgb_distance
    )

    # Geometry gates substitutability, while appearance discounts geometrically
    # compatible views whose generated content no longer agrees.
    kernel = np.clip(
        geometry_similarity * appearance_similarity,
        0.0,
        1.0 - 1e-6,
    )
    np.fill_diagonal(kernel, 1.0 - 1e-6)

    # A retained frame's mass is the number of past observations it represents.
    # Including self-similarity makes this compressed density equivalent to the
    # density of the original local cluster rather than forgetting its size.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        density = kernel @ base_masses
    if not np.all(np.isfinite(density)):
        raise ValueError("density estimates must be finite")
    raw_demand_weights = base_masses * np.power(
        density + float(density_epsilon),
        -float(density_alpha),
    )
    demand_weight_sum = float(np.sum(raw_demand_weights))
    if not np.isfinite(demand_weight_sum) or demand_weight_sum <= 0:
        raise ValueError("density-balanced demand weights must have positive mass")
    demand_weights = raw_demand_weights / demand_weight_sum

    frame_to_col = {frame_idx: col for col, frame_idx in enumerate(memory_frame_indices)}
    forced_cols = [
        frame_to_col[frame_idx]
        for frame_idx in memory_frame_indices
        if frame_idx in forced_keep_frames
    ]
    selected_cols = list(forced_cols)
    selected_set = set(forced_cols)
    marginal_gains = {col: float("inf") for col in forced_cols}

    def uncovered_probability(columns):
        if not columns:
            return np.ones(len(memory_frame_indices), dtype=np.float64)
        return np.exp(
            np.sum(np.log1p(-kernel[:, columns]), axis=1)
        )

    uncovered = uncovered_probability(selected_cols)

    selected_limit = min(int(budget), len(memory_frame_indices))
    while len(selected_cols) < selected_limit:
        gains = np.sum(
            demand_weights[:, None] * uncovered[:, None] * kernel,
            axis=0,
        )
        if selected_set:
            gains[list(selected_set)] = -np.inf
        best_col = int(np.argmax(gains))
        if not np.isfinite(gains[best_col]):
            break

        selected_cols.append(best_col)
        selected_set.add(best_col)
        marginal_gains[best_col] = float(gains[best_col])
        uncovered *= 1.0 - kernel[:, best_col]

    selected_frames = [memory_frame_indices[col] for col in selected_cols]
    selected_frame_set = set(selected_frames)
    selected_ranks = {col: rank for rank, col in enumerate(selected_cols)}
    coverage_probability = 1.0 - uncovered
    coverage_value = float(np.sum(demand_weights * coverage_probability))

    representative_masses = {col: 0.0 for col in selected_cols}
    if len(selected_cols) == len(memory_frame_indices):
        representative_masses = {
            col: float(base_masses[col]) for col in selected_cols
        }
    elif selected_cols:
        assignments = np.argmax(kernel[:, selected_cols], axis=1)
        for row, selected_position in enumerate(assignments):
            selected_col = selected_cols[int(selected_position)]
            representative_masses[selected_col] += float(base_masses[row])

    removal_losses = {}
    for col in selected_cols:
        other_cols = [other for other in selected_cols if other != col]
        uncovered_without_col = uncovered_probability(other_cols)
        removal_losses[col] = float(
            np.sum(
                demand_weights
                * kernel[:, col]
                * uncovered_without_col
            )
        )

    scores = {}
    details = {}
    for col, frame_idx in enumerate(memory_frame_indices):
        selected = frame_idx in selected_frame_set
        forced = frame_idx in forced_keep_frames
        if forced:
            score = float("inf")
        elif selected:
            score = 1.0 + removal_losses.get(col, 0.0)
        else:
            score = -1.0

        candidate_gain = (
            0.0
            if selected
            else float(
                np.sum(demand_weights * uncovered * kernel[:, col])
            )
        )
        gain = marginal_gains.get(col)
        scores[frame_idx] = float(score)
        details[frame_idx] = {
            "score": float(score),
            "density_coverage_selected": bool(selected),
            "density_coverage_forced_keep": bool(forced),
            "density_coverage_rank": selected_ranks.get(col),
            "density_coverage_marginal_gain": (
                None if gain is None or np.isinf(gain) else float(gain)
            ),
            "density_coverage_candidate_gain": candidate_gain,
            "density_coverage_removal_loss": (
                float(removal_losses.get(col, 0.0)) if selected else 0.0
            ),
            "density_coverage_base_mass": float(base_masses[col]),
            "density_coverage_assigned_mass": float(
                representative_masses.get(col, 0.0)
            ),
            "density_coverage_total_mass": total_mass,
            "density_coverage_density": float(density[col]),
            "density_coverage_demand_weight": float(demand_weights[col]),
            "density_coverage_value": coverage_value,
            "density_coverage_mean": coverage_value,
            "density_coverage_min": float(np.min(coverage_probability)),
            "density_coverage_p10": float(
                np.quantile(coverage_probability, 0.10)
            ),
            "density_coverage_geometry_mean": float(
                np.mean(geometry_similarity[:, col])
            ),
            "density_coverage_dino_mean": float(
                np.mean(dino_similarity[:, col])
            ),
            "density_coverage_rgb_distance_mean": float(
                np.mean(rgb_distance[:, col])
            ),
            "density_coverage_kernel_mean": float(np.mean(kernel[:, col])),
            "density_coverage_alpha": float(density_alpha),
            "density_coverage_dino_weight": float(dino_weight),
            "density_coverage_rgb_weight": float(rgb_weight),
            "density_coverage_epsilon": float(density_epsilon),
        }

    return (scores, details) if return_details else scores


def compute_future_view_coverage_scores(
    memory_frame_indices,
    c2ws,
    budget,
    future_query_frame_indices=None,
    forced_keep_frames=None,
    dino_features=None,
    rgb_features=None,
    coverage_masses=None,
    density_alpha=0.5,
    dino_weight=0.5,
    rgb_weight=0.25,
    future_query_weight=1.0,
    density_epsilon=1e-3,
    fov_half_h=45.0,
    fov_half_v=30.0,
    radius=50.0,
    return_details=False,
):
    """Density-balanced view coverage with an explicit prospective-demand term.

    ``compute_density_balanced_view_coverage_scores`` treats the retained bank
    plus new frames as *both* the candidate pool and the query/demand pool: a
    frame is worth keeping only if it is not yet well-covered by other
    retained frames. That makes retention self-referential -- nothing pulls a
    frame forward in time to serve a viewpoint the trajectory has not reached
    yet, so retained mass drifts toward whatever already looks locally
    redundant "now" rather than toward frames a long-gap revisit will need
    later.

    This variant adds a second, decoupled demand pool built from the known
    future camera path (``future_query_frame_indices``, e.g. the manifest's
    remaining ``c2ws`` positions ahead of the current generation point).
    Because a future query's visual content has not been generated yet, only
    the geometric FOV-overlap kernel is defined for it -- there is no
    DINO/RGB signal to discount it with. Both demand pools are combined into
    one noisy-OR coverage objective

        U(M) = sum_q w_q * (1 - prod_{x in M} (1 - K(q, x)))

    and the bank is built by forward greedy maximization of U, same as the
    self-covering policy (monotone submodular set coverage; greedy attains
    the Nemhauser et al. 1978 (1 - 1/e) guarantee). Passing no future queries
    reduces this exactly to ``compute_density_balanced_view_coverage_scores``.
    """
    memory_frame_indices = list(memory_frame_indices)
    forced_keep_frames = set(forced_keep_frames or [])
    future_query_frame_indices = [
        int(frame_idx)
        for frame_idx in dict.fromkeys(future_query_frame_indices or [])
    ]
    if budget is None:
        raise ValueError("future_view_coverage requires an explicit memory budget")
    if budget <= 0:
        raise ValueError("future_view_coverage budget must be positive")
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}
    if len(set(memory_frame_indices)) != len(memory_frame_indices):
        raise ValueError("future_view_coverage candidates must be unique")
    if dino_features is None or rgb_features is None:
        raise ValueError("future_view_coverage requires DINO and RGB features")
    if not 0.0 <= density_alpha <= 1.0:
        raise ValueError("density_alpha must be in [0, 1]")
    if dino_weight < 0 or rgb_weight < 0:
        raise ValueError("density kernel weights must be non-negative")
    if future_query_weight < 0:
        raise ValueError("future_query_weight must be non-negative")
    if density_epsilon <= 0:
        raise ValueError("density_epsilon must be positive")

    candidate_set = set(memory_frame_indices)
    unknown_forced = forced_keep_frames - candidate_set
    if unknown_forced:
        raise ValueError(
            "future_view_coverage forced-keep frames are not candidates: "
            f"{sorted(unknown_forced)[:10]}"
        )
    if len(forced_keep_frames) > budget:
        raise ValueError("future_view_coverage has more forced frames than its budget")

    missing_dino = [
        frame_idx for frame_idx in memory_frame_indices if frame_idx not in dino_features
    ]
    missing_rgb = [
        frame_idx for frame_idx in memory_frame_indices if frame_idx not in rgb_features
    ]
    if missing_dino or missing_rgb:
        raise ValueError(
            "Missing future view coverage features for frames: "
            f"DINO={missing_dino[:10]}, RGB={missing_rgb[:10]}"
        )

    coverage_masses = coverage_masses or {}
    base_masses = np.asarray(
        [coverage_masses.get(frame_idx, 1.0) for frame_idx in memory_frame_indices],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(base_masses)) or np.any(base_masses < 0):
        raise ValueError("coverage masses must be finite and non-negative")
    total_mass = float(np.sum(base_masses))
    if total_mass <= 0:
        raise ValueError("coverage masses must have positive total mass")

    geometry_similarity = camera_trajectory_similarity(
        c2ws=c2ws,
        query_frame_indices=memory_frame_indices,
        memory_frame_indices=memory_frame_indices,
        fov_half_h=fov_half_h,
        fov_half_v=fov_half_v,
        radius=radius,
    )
    dino_similarity = _feature_cosine_similarity(memory_frame_indices, dino_features)
    dino_distance = np.clip(1.0 - dino_similarity, 0.0, 2.0)
    rgb_matrix = np.stack([rgb_features[frame_idx] for frame_idx in memory_frame_indices])
    rgb_distance = pairwise_mean_abs_distances(rgb_matrix)
    if not np.all(np.isfinite(rgb_distance)):
        raise ValueError("RGB memory features must be finite")
    appearance_similarity = np.exp(
        -float(dino_weight) * dino_distance - float(rgb_weight) * rgb_distance
    )

    self_kernel = np.clip(geometry_similarity * appearance_similarity, 0.0, 1.0 - 1e-6)
    np.fill_diagonal(self_kernel, 1.0 - 1e-6)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        density = self_kernel @ base_masses
    if not np.all(np.isfinite(density)):
        raise ValueError("density estimates must be finite")
    self_demand_raw = base_masses * np.power(density + float(density_epsilon), -float(density_alpha))

    future_query_frame_indices = [
        frame_idx for frame_idx in future_query_frame_indices if frame_idx not in candidate_set
    ]
    if future_query_frame_indices:
        future_geometry = camera_trajectory_similarity(
            c2ws=c2ws,
            query_frame_indices=future_query_frame_indices,
            memory_frame_indices=memory_frame_indices,
            fov_half_h=fov_half_h,
            fov_half_v=fov_half_v,
            radius=radius,
        )
        future_kernel = np.clip(future_geometry, 0.0, 1.0 - 1e-6)
        future_demand_raw = np.full(
            len(future_query_frame_indices), float(future_query_weight), dtype=np.float64
        )
    else:
        future_kernel = np.zeros((0, len(memory_frame_indices)), dtype=np.float64)
        future_demand_raw = np.zeros(0, dtype=np.float64)

    kernel = np.vstack([self_kernel, future_kernel])
    raw_demand_weights = np.concatenate([self_demand_raw, future_demand_raw])
    demand_weight_sum = float(np.sum(raw_demand_weights))
    if not np.isfinite(demand_weight_sum) or demand_weight_sum <= 0:
        raise ValueError("future view coverage demand weights must have positive mass")
    demand_weights = raw_demand_weights / demand_weight_sum
    num_self_rows = len(memory_frame_indices)

    frame_to_col = {frame_idx: col for col, frame_idx in enumerate(memory_frame_indices)}
    forced_cols = [
        frame_to_col[frame_idx]
        for frame_idx in memory_frame_indices
        if frame_idx in forced_keep_frames
    ]
    selected_cols = list(forced_cols)
    selected_set = set(forced_cols)
    marginal_gains = {col: float("inf") for col in forced_cols}

    def uncovered_probability(columns):
        if not columns:
            return np.ones(kernel.shape[0], dtype=np.float64)
        return np.exp(np.sum(np.log1p(-kernel[:, columns]), axis=1))

    uncovered = uncovered_probability(selected_cols)

    selected_limit = min(int(budget), len(memory_frame_indices))
    while len(selected_cols) < selected_limit:
        gains = np.sum(demand_weights[:, None] * uncovered[:, None] * kernel, axis=0)
        if selected_set:
            gains[list(selected_set)] = -np.inf
        best_col = int(np.argmax(gains))
        if not np.isfinite(gains[best_col]):
            break

        selected_cols.append(best_col)
        selected_set.add(best_col)
        marginal_gains[best_col] = float(gains[best_col])
        uncovered *= 1.0 - kernel[:, best_col]

    selected_frame_set = {memory_frame_indices[col] for col in selected_cols}
    selected_ranks = {col: rank for rank, col in enumerate(selected_cols)}
    coverage_probability = 1.0 - uncovered
    coverage_value = float(np.sum(demand_weights * coverage_probability))
    future_coverage_value = (
        float(np.sum(demand_weights[num_self_rows:] * coverage_probability[num_self_rows:]))
        if future_query_frame_indices
        else 0.0
    )

    removal_losses = {}
    for col in selected_cols:
        other_cols = [other for other in selected_cols if other != col]
        uncovered_without_col = uncovered_probability(other_cols)
        removal_losses[col] = float(
            np.sum(demand_weights * kernel[:, col] * uncovered_without_col)
        )

    # Mass conservation mirrors compute_density_balanced_view_coverage_scores:
    # only the self-covering rows carry a retained "number of past observations"
    # mass to redistribute across sections. Future query rows are synthetic
    # viewpoints, not observations, so they never contribute or receive mass.
    representative_masses = {col: 0.0 for col in selected_cols}
    if len(selected_cols) == num_self_rows:
        representative_masses = {col: float(base_masses[col]) for col in selected_cols}
    elif selected_cols:
        self_assignments = np.argmax(self_kernel[:, selected_cols], axis=1)
        for row, selected_position in enumerate(self_assignments):
            selected_col = selected_cols[int(selected_position)]
            representative_masses[selected_col] += float(base_masses[row])

    scores = {}
    details = {}
    for col, frame_idx in enumerate(memory_frame_indices):
        selected = frame_idx in selected_frame_set
        forced = frame_idx in forced_keep_frames
        if forced:
            score = float("inf")
        elif selected:
            score = 1.0 + removal_losses.get(col, 0.0)
        else:
            score = -1.0

        candidate_gain = (
            0.0
            if selected
            else float(np.sum(demand_weights * uncovered * kernel[:, col]))
        )
        gain = marginal_gains.get(col)
        scores[frame_idx] = float(score)
        details[frame_idx] = {
            "score": float(score),
            "future_view_coverage_selected": bool(selected),
            "future_view_coverage_forced_keep": bool(forced),
            "future_view_coverage_rank": selected_ranks.get(col),
            "future_view_coverage_marginal_gain": (
                None if gain is None or np.isinf(gain) else float(gain)
            ),
            "future_view_coverage_candidate_gain": candidate_gain,
            "future_view_coverage_removal_loss": (
                float(removal_losses.get(col, 0.0)) if selected else 0.0
            ),
            "future_view_coverage_assigned_mass": float(
                representative_masses.get(col, 0.0)
            ),
            "future_view_coverage_base_mass": float(base_masses[col]),
            "future_view_coverage_total_mass": total_mass,
            "future_view_coverage_value": coverage_value,
            "future_view_coverage_future_value": future_coverage_value,
            "future_view_coverage_future_query_count": len(future_query_frame_indices),
            "future_view_coverage_future_kernel_mean": (
                float(np.mean(future_kernel[:, col])) if future_query_frame_indices else 0.0
            ),
            "future_view_coverage_alpha": float(density_alpha),
            "future_view_coverage_dino_weight": float(dino_weight),
            "future_view_coverage_rgb_weight": float(rgb_weight),
            "future_view_coverage_future_query_weight": float(future_query_weight),
        }

    return (scores, details) if return_details else scores


def _historical_query_medoids(memory_frame_indices, dino_features, rarity_neighbors=3):
    """Cluster candidates by DINO similarity and return one medoid per cluster.

    Mirrors the clustering already used by ``compute_rarity_irreplaceability_scores``
    so historical demand is grounded in the same notion of "distinct scene mode".
    Each cluster gets one query point, regardless of cluster size -- this is the
    "controlled rather than frequency-proportional" weighting: a corridor visited
    100 times is one query, same as a room visited once.
    """
    memory_frame_indices = list(memory_frame_indices)
    if len(memory_frame_indices) == 1:
        return [memory_frame_indices[0]], [[0]]

    dino_matrix = np.stack([dino_features[frame_idx] for frame_idx in memory_frame_indices])
    dino_pairwise = cosine_distances(dino_matrix)
    np.fill_diagonal(dino_pairwise, np.inf)
    threshold = estimate_cluster_threshold(dino_pairwise, rarity_neighbors)

    cluster_pairwise = dino_pairwise.copy()
    np.fill_diagonal(cluster_pairwise, 0.0)
    _, clusters = connected_components_from_threshold(cluster_pairwise, threshold=threshold)

    medoid_positions = []
    for members in clusters:
        if len(members) == 1:
            medoid_positions.append(members[0])
            continue
        sub_distances = cluster_pairwise[np.ix_(members, members)]
        total_distance = sub_distances.sum(axis=1)
        medoid_positions.append(members[int(np.argmin(total_distance))])

    medoid_frame_indices = [memory_frame_indices[position] for position in medoid_positions]
    return medoid_frame_indices, clusters


def compute_marginal_coverage_eviction_scores(
    memory_frame_indices,
    c2ws,
    budget,
    future_query_frame_indices=None,
    forced_keep_frames=None,
    dino_features=None,
    rgb_features=None,
    alpha=0.65,
    lambda_hist=None,
    gamma=0.25,
    hist_freq_bias=0.0,
    rarity_neighbors=3,
    fov_half_h=45.0,
    fov_half_v=30.0,
    radius=50.0,
    return_details=False,
):
    """Marginal Coverage Eviction (MCE): the paper-faithful set-coverage policy.

    Implements the method as specified, not an approximation of it:

    - Query set Q = Q_hist ∪ Q_ctrl (Sec. 3.1). Q_hist is one DINO-cluster
      medoid per distinct scene mode among the candidates. Each medoid's
      weight is proportional to (cluster_size ** hist_freq_bias):
      hist_freq_bias=0 (default, paper's stated rule) weights every medoid
      equally regardless of cluster size, so a repeatedly revisited region
      cannot outweigh a rare one; hist_freq_bias=1 weights medoids linearly
      by how often their scene mode recurs, closer to what an
      anchor-persistence heuristic like SLAM implicitly rewards. This is an
      explicit generalization knob, not a paper claim -- exposed to test
      whether MCE can reach/exceed SLAM's regime by relaxing the
      diversity-favoring default toward frequency-favoring, rather than
      assuming the two are unrelated methods.
      Q_ctrl is the known future camera path, sampled and weighted with
      exponential horizon decay ``w_h ∝ exp(-gamma * h)``; omitted when no
      future controls are known (``lambda_hist`` then defaults to 1).
    - Kernel K(q, m) = alpha * K_geo(q, m) + (1 - alpha) * K_vis(q, m), an
      explicit convex combination (Eq. 6), not a product. Future queries have
      no realized appearance yet, so they use K_geo alone -- "the available
      cue is used alone".
    - Eviction is reverse deletion (Algorithm 1): repeatedly remove
      argmin_i Delta_i(P) from the full candidate pool P, recomputing exact
      deletion marginals after each removal, until |P| <= B. This is
      deliberately not forward-greedy addition -- the paper does not claim
      the offline forward-greedy (1 - 1/e) guarantee transfers to this rule.
    """
    memory_frame_indices = list(memory_frame_indices)
    forced_keep_frames = set(forced_keep_frames or [])
    future_query_frame_indices = [
        int(frame_idx)
        for frame_idx in dict.fromkeys(future_query_frame_indices or [])
        if frame_idx not in set(memory_frame_indices)
    ]
    if budget is None:
        raise ValueError("mce requires an explicit memory budget")
    if budget <= 0:
        raise ValueError("mce budget must be positive")
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}
    if len(set(memory_frame_indices)) != len(memory_frame_indices):
        raise ValueError("mce candidates must be unique")
    if dino_features is None or rgb_features is None:
        raise ValueError("mce requires DINO and RGB features")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("mce alpha must be in [0, 1]")
    if lambda_hist is not None and not 0.0 <= lambda_hist <= 1.0:
        raise ValueError("mce lambda_hist must be in [0, 1]")
    if gamma < 0:
        raise ValueError("mce gamma must be non-negative")

    candidate_set = set(memory_frame_indices)
    unknown_forced = forced_keep_frames - candidate_set
    if unknown_forced:
        raise ValueError(f"mce forced-keep frames are not candidates: {sorted(unknown_forced)[:10]}")
    if len(forced_keep_frames) > budget:
        raise ValueError("mce has more forced frames than its budget")

    missing_dino = [f for f in memory_frame_indices if f not in dino_features]
    missing_rgb = [f for f in memory_frame_indices if f not in rgb_features]
    if missing_dino or missing_rgb:
        raise ValueError(f"Missing mce features for frames: DINO={missing_dino[:10]}, RGB={missing_rgb[:10]}")

    num_candidates = len(memory_frame_indices)
    selected_limit = min(int(budget), num_candidates)

    # --- Query set construction (Sec. 3.1) ---------------------------------
    hist_query_frame_indices, hist_clusters = _historical_query_medoids(
        memory_frame_indices, dino_features, rarity_neighbors
    )
    num_hist = len(hist_query_frame_indices)
    lambda_eff = (
        1.0
        if not future_query_frame_indices
        else (0.5 if lambda_hist is None else float(lambda_hist))
    )

    # --- Kernel (Eq. 6): explicit convex combination, not a product --------
    hist_geo = camera_trajectory_similarity(
        c2ws=c2ws,
        query_frame_indices=hist_query_frame_indices,
        memory_frame_indices=memory_frame_indices,
        fov_half_h=fov_half_h,
        fov_half_v=fov_half_v,
        radius=radius,
    )
    hist_vis_cosine = _feature_cosine_similarity_cross(
        hist_query_frame_indices, memory_frame_indices, dino_features
    )
    hist_vis = np.clip((hist_vis_cosine + 1.0) / 2.0, 0.0, 1.0)  # calibrate [-1,1] -> [0,1]
    hist_kernel = np.clip(alpha * hist_geo + (1.0 - alpha) * hist_vis, 0.0, 1.0 - 1e-6)
    cluster_sizes = np.array([len(members) for members in hist_clusters], dtype=np.float64)
    raw_hist_weights = np.power(np.maximum(cluster_sizes, 1.0), float(hist_freq_bias))
    hist_weights = lambda_eff * raw_hist_weights / max(np.sum(raw_hist_weights), 1e-12)

    if future_query_frame_indices:
        num_ctrl = len(future_query_frame_indices)
        ctrl_geo = camera_trajectory_similarity(
            c2ws=c2ws,
            query_frame_indices=future_query_frame_indices,
            memory_frame_indices=memory_frame_indices,
            fov_half_h=fov_half_h,
            fov_half_v=fov_half_v,
            radius=radius,
        )
        ctrl_kernel = np.clip(ctrl_geo, 0.0, 1.0 - 1e-6)
        horizons = np.arange(1, num_ctrl + 1, dtype=np.float64)
        raw_ctrl_weights = np.exp(-float(gamma) * horizons)
        ctrl_weights = (1.0 - lambda_eff) * raw_ctrl_weights / np.sum(raw_ctrl_weights)
    else:
        ctrl_kernel = np.zeros((0, num_candidates), dtype=np.float64)
        ctrl_weights = np.zeros(0, dtype=np.float64)

    kernel = np.vstack([hist_kernel, ctrl_kernel])
    weights = np.concatenate([hist_weights, ctrl_weights])
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("mce query weights must have positive total mass")
    weights = weights / weight_sum

    # --- Algorithm 1: reverse deletion with exact recomputed marginals -----
    frame_to_col = {frame_idx: col for col, frame_idx in enumerate(memory_frame_indices)}
    forced_cols = {frame_to_col[frame_idx] for frame_idx in forced_keep_frames}
    remaining_cols = list(range(num_candidates))
    one_minus_kernel = 1.0 - kernel
    pool_product = np.prod(one_minus_kernel, axis=1)  # P_q over the full initial pool

    removal_order = []
    removal_marginals = {}
    while len(remaining_cols) > selected_limit:
        remaining = np.array(remaining_cols)
        denom = np.maximum(one_minus_kernel[:, remaining], 1e-12)
        marginals = np.sum(
            (weights[:, None] * kernel[:, remaining] * pool_product[:, None]) / denom,
            axis=0,
        )
        eviction_candidates = [
            (float(marginals[position]), col)
            for position, col in enumerate(remaining_cols)
            if col not in forced_cols
        ]
        if not eviction_candidates:
            break
        loss, evict_col = min(eviction_candidates, key=lambda item: item[0])
        removal_order.append(evict_col)
        removal_marginals[evict_col] = loss
        pool_product = pool_product / np.maximum(one_minus_kernel[:, evict_col], 1e-12)
        remaining_cols.remove(evict_col)

    selected_cols = remaining_cols
    selected_frame_set = {memory_frame_indices[col] for col in selected_cols}

    # Final leave-one-out marginal for each survivor, w.r.t. the final set --
    # this is the reported "value", not what drove the eviction decisions.
    final_uncovered = np.prod(one_minus_kernel[:, selected_cols], axis=1) if selected_cols else np.ones(kernel.shape[0])
    survivor_marginals = {}
    for col in selected_cols:
        denom = np.maximum(one_minus_kernel[:, col], 1e-12)
        survivor_marginals[col] = float(
            np.sum(weights * kernel[:, col] * final_uncovered / denom)
        )
    coverage_value = float(np.sum(weights * (1.0 - final_uncovered)))

    scores = {}
    details = {}
    for col, frame_idx in enumerate(memory_frame_indices):
        selected = frame_idx in selected_frame_set
        forced = frame_idx in forced_keep_frames
        if forced:
            score = float("inf")
        elif selected:
            score = 1.0 + survivor_marginals.get(col, 0.0)
        else:
            score = -1.0 - removal_marginals.get(col, 0.0)

        scores[frame_idx] = float(score)
        details[frame_idx] = {
            "score": float(score),
            "mce_selected": bool(selected),
            "mce_forced_keep": bool(forced),
            "mce_removal_rank": (
                removal_order.index(col) if col in removal_order else None
            ),
            "mce_removal_marginal": removal_marginals.get(col),
            "mce_survivor_marginal": survivor_marginals.get(col),
            "mce_coverage_value": coverage_value,
            "mce_alpha": float(alpha),
            "mce_lambda": float(lambda_eff),
            "mce_gamma": float(gamma),
            "mce_hist_freq_bias": float(hist_freq_bias),
            "mce_rarity_neighbors": int(rarity_neighbors),
            "mce_num_hist_queries": num_hist,
            "mce_num_ctrl_queries": len(future_query_frame_indices),
            "mce_hist_query_frames": [int(f) for f in hist_query_frame_indices],
        }

    return (scores, details) if return_details else scores


def compute_slam_ri_blend_scores(
    memory_frame_indices,
    c2ws,
    forced_keep_frames=None,
    dino_features=None,
    rgb_features=None,
    beta=0.5,
    slam_kwargs=None,
    ri_kwargs=None,
    return_details=False,
):
    """Linear blend of SLAM's and RI's own scores: V_i = beta*S~_i + (1-beta)*RI~_i.

    S_i is the unmodified output of compute_slam_covisibility_scores and RI_i
    is the unmodified output of compute_rarity_irreplaceability_scores -- this
    function calls both directly rather than reimplementing either, so
    beta=1.0 exactly reproduces slam_covisibility's own ranking and beta=0.0
    exactly reproduces rarity_irreplaceability's own ranking. Both functions
    are called with pinned_frames=None so forced-keep handling happens once,
    here, after blending (avoids each sub-score independently producing inf
    and washing out the min-max normalization below).

    The two raw scores live on unrelated scales -- SLAM's is a bounded sum of
    three roughly-[0,1] terms; RI's is log-rarity times an RGB L1 distance,
    unbounded above -- so each is min-max normalized to [0, 1] over the
    current candidate pool before blending.

    Neither constituent score is budget-aware or iterative (no reverse-
    deletion/marginal-gain loop, unlike MCE), so this is a flat scorer like
    its two parents: FrameMemoryBuffer.evict_to_budget() performs the actual
    min-score-first eviction externally.
    """
    memory_frame_indices = list(memory_frame_indices)
    forced_keep_frames = set(forced_keep_frames or ())
    slam_kwargs = dict(slam_kwargs or {})
    ri_kwargs = dict(ri_kwargs or {})
    beta = float(beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}

    slam_scores, slam_details = compute_slam_covisibility_scores(
        memory_frame_indices=memory_frame_indices,
        c2ws=c2ws,
        pinned_frames=None,
        dino_features=dino_features,
        rgb_features=rgb_features,
        return_details=True,
        **slam_kwargs,
    )
    ri_scores, ri_details = compute_rarity_irreplaceability_scores(
        memory_frame_indices=memory_frame_indices,
        pinned_frames=None,
        dino_features=dino_features,
        rgb_features=rgb_features,
        return_details=True,
        **ri_kwargs,
    )

    def _min_max_normalize(raw):
        values = np.array([raw[idx] for idx in memory_frame_indices], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {idx: 0.0 for idx in memory_frame_indices}
        lo, hi = float(finite.min()), float(finite.max())
        span = hi - lo
        normalized = {}
        for idx, value in zip(memory_frame_indices, values):
            if not np.isfinite(value):
                normalized[idx] = 1.0
            elif span <= 1e-12:
                normalized[idx] = 1.0
            else:
                normalized[idx] = (float(value) - lo) / span
        return normalized

    slam_norm = _min_max_normalize(slam_scores)
    ri_norm = _min_max_normalize(ri_scores)

    scores = {}
    details = {}
    for idx in memory_frame_indices:
        is_forced = idx in forced_keep_frames
        score = float("inf") if is_forced else beta * slam_norm[idx] + (1.0 - beta) * ri_norm[idx]
        scores[idx] = score
        if return_details:
            details[idx] = {
                "score": score,
                "slamri_beta": beta,
                "slamri_forced_keep": is_forced,
                "slamri_slam_raw": float(slam_scores[idx]),
                "slamri_slam_norm": float(slam_norm[idx]),
                "slamri_ri_raw": float(ri_scores[idx]),
                "slamri_ri_norm": float(ri_norm[idx]),
                "slamri_ri_rarity": ri_details[idx]["rarity"],
                "slamri_ri_irreplaceability": ri_details[idx]["irreplaceability"],
                "slamri_slam_redundancy_ratio": slam_details[idx]["redundancy_ratio"],
                "slamri_slam_unique_bonus": slam_details[idx]["unique_bonus"],
            }

    return (scores, details) if return_details else scores


def compute_trajectory_coverage_scores(
    memory_frame_indices,
    archive_frame_indices,
    c2ws,
    budget,
    forced_keep_frames=None,
    archive_weights=None,
    fov_half_h=45.0,
    fov_half_v=30.0,
    radius=50.0,
    return_details=False,
):
    """Select memories that maximize coverage of the observed camera trajectory.

    For every past trajectory pose, coverage is the best direct camera-FOV
    similarity supplied by any selected memory. Greedy facility-location
    selection gives the standard diminishing-returns coreset and only receives
    archive indices at or before the current generation section.
    """
    memory_frame_indices = list(memory_frame_indices)
    archive_frame_indices = list(archive_frame_indices)
    forced_keep_frames = set(forced_keep_frames or [])
    if budget is None:
        raise ValueError("trajectory_coverage requires an explicit memory budget")
    if budget <= 0:
        raise ValueError("trajectory_coverage budget must be positive")
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}
    if not archive_frame_indices:
        archive_frame_indices = list(memory_frame_indices)

    unknown_forced = forced_keep_frames - set(memory_frame_indices)
    if unknown_forced:
        raise ValueError(
            "trajectory_coverage forced-keep frames are not candidates: "
            f"{sorted(unknown_forced)[:10]}"
        )
    if len(forced_keep_frames) > budget:
        raise ValueError("trajectory_coverage has more forced frames than its budget")

    weights = (
        np.ones(len(archive_frame_indices), dtype=np.float64)
        if archive_weights is None
        else np.asarray(archive_weights, dtype=np.float64)
    )
    if weights.shape != (len(archive_frame_indices),):
        raise ValueError("archive_weights length must match archive_frame_indices")
    if np.any(weights < 0):
        raise ValueError("archive_weights must be non-negative")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        raise ValueError("trajectory_coverage archive weights must have positive mass")
    weights = weights / weight_sum

    similarity = camera_trajectory_similarity(
        c2ws=c2ws,
        query_frame_indices=archive_frame_indices,
        memory_frame_indices=memory_frame_indices,
        fov_half_h=fov_half_h,
        fov_half_v=fov_half_v,
        radius=radius,
    )

    frame_to_col = {frame_idx: col for col, frame_idx in enumerate(memory_frame_indices)}
    forced_cols = [
        frame_to_col[frame_idx]
        for frame_idx in memory_frame_indices
        if frame_idx in forced_keep_frames
    ]
    selected_cols = list(forced_cols)
    selected_set = set(forced_cols)
    marginal_gains = {col: float("inf") for col in forced_cols}

    if selected_cols:
        covered = np.max(similarity[:, selected_cols], axis=1)
    else:
        covered = np.zeros(len(archive_frame_indices), dtype=np.float64)

    selected_limit = min(int(budget), len(memory_frame_indices))
    while len(selected_cols) < selected_limit:
        gains = np.sum(
            weights[:, None] * np.maximum(similarity - covered[:, None], 0.0),
            axis=0,
        )
        if selected_set:
            gains[list(selected_set)] = -np.inf
        best_col = int(np.argmax(gains))
        if not np.isfinite(gains[best_col]):
            break

        selected_cols.append(best_col)
        selected_set.add(best_col)
        marginal_gains[best_col] = float(gains[best_col])
        covered = np.maximum(covered, similarity[:, best_col])

    selected_frames = [memory_frame_indices[col] for col in selected_cols]
    selected_frame_set = set(selected_frames)
    trajectory_value = float(np.sum(weights * covered))

    removal_losses = {}
    for col in selected_cols:
        other_cols = [other for other in selected_cols if other != col]
        without_col = (
            np.max(similarity[:, other_cols], axis=1)
            if other_cols
            else np.zeros(len(archive_frame_indices), dtype=np.float64)
        )
        removal_losses[col] = float(np.sum(weights * (covered - without_col)))

    scores = {}
    details = {}
    for col, frame_idx in enumerate(memory_frame_indices):
        selected = frame_idx in selected_frame_set
        forced = frame_idx in forced_keep_frames
        if forced:
            score = float("inf")
        elif selected:
            score = 1.0 + removal_losses.get(col, 0.0)
        else:
            score = -1.0

        gain = marginal_gains.get(col)
        candidate_gain = float(
            np.sum(weights * np.maximum(similarity[:, col] - covered, 0.0))
        )
        scores[frame_idx] = float(score)
        details[frame_idx] = {
            "score": float(score),
            "trajectory_selected": bool(selected),
            "trajectory_forced_keep": bool(forced),
            "trajectory_rank": selected_frames.index(frame_idx) if selected else None,
            "trajectory_marginal_gain": (
                None if gain is None or np.isinf(gain) else float(gain)
            ),
            "trajectory_candidate_gain": candidate_gain,
            "trajectory_removal_loss": (
                float(removal_losses.get(col, 0.0)) if selected else 0.0
            ),
            "trajectory_archive_size": len(archive_frame_indices),
            "trajectory_value": trajectory_value,
            "trajectory_coverage_mean": float(np.mean(covered)),
            "trajectory_coverage_min": float(np.min(covered)),
            "trajectory_coverage_p10": float(np.quantile(covered, 0.10)),
            "trajectory_similarity_mean": float(np.mean(similarity[:, col])),
            "trajectory_similarity_max": float(np.max(similarity[:, col])),
            "trajectory_fov_half_h": float(fov_half_h),
            "trajectory_fov_half_v": float(fov_half_v),
            "trajectory_radius": float(radius),
        }

    return (scores, details) if return_details else scores


def compute_kcenter_coreset_scores(
    memory_frame_indices,
    archive_frame_indices,
    c2ws,
    budget,
    forced_keep_frames=None,
    dino_features=None,
    visual_weight=0.5,
    pose_weight=0.5,
    time_weight=0.0,
    return_details=False,
):
    """Select memory frames by active-learning-style k-center coverage.

    The selected set is built greedily by repeatedly finding the archive frame
    currently farthest from any selected center, then adding the candidate memory
    frame nearest to that farthest archive point. Higher returned scores mean
    "keep"; unselected frames get low scores and are evicted by the buffer.
    """
    memory_frame_indices = list(memory_frame_indices)
    archive_frame_indices = list(archive_frame_indices)
    forced_keep_frames = set(forced_keep_frames or [])
    if budget is None:
        raise ValueError("kcenter_coreset requires an explicit memory budget")
    if budget <= 0:
        raise ValueError("kcenter_coreset budget must be positive")
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}
    if not archive_frame_indices:
        archive_frame_indices = list(memory_frame_indices)

    use_visual = dino_features is not None and float(visual_weight) > 0.0
    if use_visual:
        missing_features = [
            frame_idx
            for frame_idx in set(memory_frame_indices) | set(archive_frame_indices)
            if frame_idx not in dino_features
        ]
        if missing_features:
            raise ValueError(f"Missing k-center DINO features for frames: {missing_features[:10]}")

    if len(memory_frame_indices) <= budget:
        scores = {
            frame_idx: float("inf") if frame_idx in forced_keep_frames else 1.0
            for frame_idx in memory_frame_indices
        }
        details = {
            frame_idx: {
                "score": scores[frame_idx],
                "kcenter_selected": True,
                "kcenter_forced_keep": frame_idx in forced_keep_frames,
                "kcenter_rank": index,
                "kcenter_radius": 0.0,
                "kcenter_current_radius": 0.0,
                "kcenter_removal_radius_increase": None,
                "kcenter_archive_size": len(archive_frame_indices),
            }
            for index, frame_idx in enumerate(memory_frame_indices)
        }
        return (scores, details) if return_details else scores

    components = []
    if use_visual:
        visual_similarity = _feature_cosine_similarity_cross(
            archive_frame_indices,
            memory_frame_indices,
            dino_features,
        )
        visual_distance = (1.0 - visual_similarity) / 2.0
        visual_distance = np.clip(visual_distance, 0.0, 1.0)
        components.append((float(visual_weight), visual_distance))

    if pose_weight:
        pose_distance = pose_distances(c2ws, archive_frame_indices, memory_frame_indices)
        pose_distance = 1.0 - np.exp(-pose_distance)
        components.append((float(pose_weight), pose_distance))

    if time_weight:
        archive_times = np.asarray(archive_frame_indices, dtype=np.float64)
        memory_times = np.asarray(memory_frame_indices, dtype=np.float64)
        time_scale = max(
            float(max(max(archive_frame_indices), max(memory_frame_indices)) + 1),
            1.0,
        )
        time_distance = np.abs(archive_times[:, None] - memory_times[None, :]) / time_scale
        components.append((float(time_weight), time_distance))

    if not components:
        raise ValueError("kcenter_coreset needs at least one positive distance component")

    total_weight = max(sum(weight for weight, _ in components), 1e-12)
    distance = sum(weight * matrix for weight, matrix in components) / total_weight

    frame_to_col = {frame_idx: col for col, frame_idx in enumerate(memory_frame_indices)}
    forced_cols = [
        frame_to_col[frame_idx]
        for frame_idx in memory_frame_indices
        if frame_idx in forced_keep_frames
    ]

    selected_cols = []
    selected_set = set()
    selected_by_archive = {}

    for col in forced_cols:
        if col not in selected_set:
            selected_set.add(col)
            selected_cols.append(col)
            selected_by_archive[col] = None

    if selected_cols:
        covered_distance = np.min(distance[:, selected_cols], axis=1)
    else:
        first_col = int(np.argmin(np.mean(distance, axis=0)))
        selected_set.add(first_col)
        selected_cols.append(first_col)
        selected_by_archive[first_col] = None
        covered_distance = distance[:, first_col].copy()

    while len(selected_cols) < min(int(budget), len(memory_frame_indices)):
        farthest_archive_row = int(np.argmax(covered_distance))
        candidate_order = np.argsort(distance[farthest_archive_row])
        best_col = None
        for col in candidate_order:
            col = int(col)
            if col not in selected_set:
                best_col = col
                break
        if best_col is None:
            break

        selected_set.add(best_col)
        selected_cols.append(best_col)
        selected_by_archive[best_col] = int(archive_frame_indices[farthest_archive_row])
        covered_distance = np.minimum(covered_distance, distance[:, best_col])

    selected_frames = [memory_frame_indices[col] for col in selected_cols]
    selected_frame_set = set(selected_frames)
    current_radius = float(np.max(covered_distance)) if covered_distance.size else 0.0
    mean_radius = float(np.mean(covered_distance)) if covered_distance.size else 0.0

    removal_radius_increases = {}
    for col in selected_cols:
        other_cols = [other for other in selected_cols if other != col]
        if other_cols:
            without_col = np.min(distance[:, other_cols], axis=1)
            without_radius = float(np.max(without_col))
        else:
            without_radius = float("inf")
        removal_radius_increases[col] = without_radius - current_radius

    scores = {}
    details = {}
    for col, frame_idx in enumerate(memory_frame_indices):
        selected = frame_idx in selected_frame_set
        forced = frame_idx in forced_keep_frames
        if forced:
            score = float("inf")
        elif selected:
            score = 1.0 + max(float(removal_radius_increases.get(col, 0.0)), 0.0)
        else:
            score = -1.0

        nearest_archive_row = int(np.argmin(distance[:, col])) if distance.shape[0] else None
        selected_archive_frame = selected_by_archive.get(col)
        rank = selected_frames.index(frame_idx) if selected else None
        scores[frame_idx] = float(score)
        details[frame_idx] = {
            "score": float(score),
            "kcenter_selected": bool(selected),
            "kcenter_forced_keep": bool(forced),
            "kcenter_rank": rank,
            "kcenter_radius": current_radius,
            "kcenter_mean_radius": mean_radius,
            "kcenter_removal_radius_increase": (
                float(removal_radius_increases.get(col, 0.0)) if selected else 0.0
            ),
            "kcenter_archive_size": len(archive_frame_indices),
            "kcenter_nearest_archive_frame": (
                None if nearest_archive_row is None else int(archive_frame_indices[nearest_archive_row])
            ),
            "kcenter_nearest_archive_distance": (
                None if nearest_archive_row is None else float(distance[nearest_archive_row, col])
            ),
            "kcenter_selected_for_archive_frame": selected_archive_frame,
            "kcenter_visual_weight": float(visual_weight if use_visual else 0.0),
            "kcenter_pose_weight": float(pose_weight),
            "kcenter_time_weight": float(time_weight),
        }

    return (scores, details) if return_details else scores


def compute_h2o_heavy_hitter_scores(
    memory_frame_indices,
    memory_stats,
    budget,
    forced_keep_frames=None,
    recency_fraction=0.25,
    count_weight=0.05,
    best_overlap_weight=0.10,
    return_details=False,
):
    """Score frames by H2O-style accumulated model read/use weight plus recency.

    The pipeline does not expose raw denoising attention over memory frames, so
    this policy uses the model's own memory retrieval decisions as the internal
    read signal: each selected context frame accumulates its selected overlap as
    attention-like mass. A fixed reserve of the newest frames is kept regardless
    of read count, matching H2O's heavy-hitter plus recent-token split.
    """
    memory_frame_indices = list(memory_frame_indices)
    forced_keep_frames = set(forced_keep_frames or [])
    if budget is None:
        raise ValueError("h2o_heavy_hitter requires an explicit memory budget")
    if budget <= 0:
        raise ValueError("h2o_heavy_hitter budget must be positive")
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}

    recency_budget = max(1, int(round(float(budget) * float(recency_fraction))))
    recency_budget = min(recency_budget, max(int(budget) - len(forced_keep_frames), 1))
    newest_frames = sorted(memory_frame_indices, reverse=True)
    recent_keep = set(newest_frames[:recency_budget])

    read_weights = np.asarray(
        [
            float(memory_stats.get(frame_idx, {}).get("selection_overlap_sum", 0.0))
            for frame_idx in memory_frame_indices
        ],
        dtype=np.float64,
    )
    selected_counts = np.asarray(
        [
            float(memory_stats.get(frame_idx, {}).get("selected_count", 0.0))
            for frame_idx in memory_frame_indices
        ],
        dtype=np.float64,
    )
    best_overlaps = np.asarray(
        [
            float(memory_stats.get(frame_idx, {}).get("best_selection_overlap", 0.0))
            for frame_idx in memory_frame_indices
        ],
        dtype=np.float64,
    )

    def scale(values):
        max_value = float(np.max(values)) if values.size else 0.0
        if max_value <= 1e-12:
            return np.zeros_like(values)
        return values / max_value

    heavy_scores = (
        scale(read_weights)
        + float(count_weight) * scale(selected_counts)
        + float(best_overlap_weight) * scale(best_overlaps)
    )
    score_by_frame = {
        frame_idx: float(heavy_scores[index])
        for index, frame_idx in enumerate(memory_frame_indices)
    }
    heavy_order = {
        frame_idx: rank
        for rank, frame_idx in enumerate(
            sorted(memory_frame_indices, key=lambda idx: (-score_by_frame[idx], -idx))
        )
    }
    recency_order = {frame_idx: rank for rank, frame_idx in enumerate(newest_frames)}

    scores = {}
    details = {}
    for index, frame_idx in enumerate(memory_frame_indices):
        forced = frame_idx in forced_keep_frames
        recent = frame_idx in recent_keep
        if forced:
            score = float("inf")
        elif recent:
            score = 1_000_000.0 + float(recency_budget - recency_order[frame_idx])
        else:
            score = float(heavy_scores[index])

        scores[frame_idx] = float(score)
        details[frame_idx] = {
            "score": float(score),
            "h2o_read_weight": float(read_weights[index]),
            "h2o_selected_count": int(selected_counts[index]),
            "h2o_best_overlap": float(best_overlaps[index]),
            "h2o_heavy_score": float(heavy_scores[index]),
            "h2o_heavy_rank": int(heavy_order[frame_idx]),
            "h2o_recent_keep": bool(recent),
            "h2o_recency_rank": int(recency_order[frame_idx]),
            "h2o_recency_budget": int(recency_budget),
            "h2o_recency_fraction": float(recency_fraction),
        }

    return (scores, details) if return_details else scores
