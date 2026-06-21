import math
from collections import OrderedDict

import numpy as np


SUPPORTED_MEMORY_POLICIES = (
    "unbounded",
    "fifo",
    "rarity_irreplaceability",
    "slam_covisibility",
)
BUDGETED_MEMORY_POLICIES = ("fifo", "rarity_irreplaceability", "slam_covisibility")


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
    finite = pairwise_distances[np.isfinite(pairwise_distances)]
    if finite.size == 0:
        return 0.0

    # Use the median nearest-neighbor distance as the mode scale. Larger k-neighbor
    # thresholds can merge a whole camera path into one chain-shaped component.
    nearest = np.partition(pairwise_distances, 0, axis=1)[:, 0]
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

    feature_matrix = np.stack([features[frame_idx] for frame_idx in memory_frame_indices])
    norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
    feature_matrix = feature_matrix / np.maximum(norms, 1e-12)
    return np.clip(feature_matrix @ feature_matrix.T, -1.0, 1.0)


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

    pose_distance = pose_distances(c2ws, memory_frame_indices, memory_frame_indices)
    geom_similarity = np.exp(-pose_distance)
    np.fill_diagonal(geom_similarity, 0.0)

    components = [(geometry_weight, geom_similarity)]
    if dino_features is not None:
        visual_similarity = _feature_cosine_similarity(memory_frame_indices, dino_features)
        visual_similarity = np.maximum(visual_similarity, 0.0)
        np.fill_diagonal(visual_similarity, 0.0)
        components.append((visual_weight, visual_similarity))
    elif rgb_features is not None:
        rgb_distance = pairwise_mean_abs_distances(
            np.stack([rgb_features[frame_idx] for frame_idx in memory_frame_indices])
        )
        rgb_similarity = np.exp(-4.0 * rgb_distance)
        np.fill_diagonal(rgb_similarity, 0.0)
        components.append((visual_weight, rgb_similarity))

    total_weight = sum(weight for weight, _ in components)
    covisibility = sum(weight * matrix for weight, matrix in components) / max(total_weight, 1e-12)
    np.fill_diagonal(covisibility, 0.0)

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
