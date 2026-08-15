import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "diffsynth" / "pipelines" / "memory_policies.py"
SPEC = importlib.util.spec_from_file_location("memory_policies", MODULE_PATH)
MEMORY_POLICIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEMORY_POLICIES)


class EstimateClusterThresholdTest(unittest.TestCase):
    def _distance_matrix(self):
        # 6 tight groups of 5 points each, well-separated between groups.
        rng = np.random.default_rng(0)
        dim = 16
        points = []
        for group in range(6):
            center = rng.normal(size=dim)
            center /= np.linalg.norm(center)
            for _ in range(5):
                vec = center + rng.normal(scale=0.01, size=dim)
                points.append(vec / np.linalg.norm(vec))
        points = np.stack(points)
        distances = MEMORY_POLICIES.cosine_distances(points)
        np.fill_diagonal(distances, np.inf)
        return distances

    def test_rarity_neighbors_actually_changes_the_threshold(self):
        # Regression test for the bug where rarity_neighbors was accepted but
        # never used -- np.partition(..., 0) always meant "1st nearest
        # neighbor" regardless of the argument. A larger k must now yield a
        # >= threshold (k-th nearest-neighbor distance is monotonically
        # non-decreasing in k), and strictly greater for this well-clustered
        # input where within-cluster and between-cluster distances differ.
        distances = self._distance_matrix()
        threshold_k1 = MEMORY_POLICIES.estimate_cluster_threshold(distances, rarity_neighbors=1)
        threshold_k4 = MEMORY_POLICIES.estimate_cluster_threshold(distances, rarity_neighbors=4)
        threshold_k20 = MEMORY_POLICIES.estimate_cluster_threshold(distances, rarity_neighbors=20)
        self.assertLess(threshold_k1, threshold_k4)
        self.assertLess(threshold_k4, threshold_k20)

    def test_rarity_neighbors_one_matches_original_nearest_neighbor_behavior(self):
        distances = self._distance_matrix()
        threshold = MEMORY_POLICIES.estimate_cluster_threshold(distances, rarity_neighbors=1)
        expected = float(np.median(np.partition(distances, 0, axis=1)[:, 0]))
        self.assertAlmostEqual(threshold, expected, places=10)

    def test_coarser_threshold_merges_clusters(self):
        # The actual point of the fix: does the corrected, larger threshold
        # at higher rarity_neighbors actually produce fewer, bigger clusters
        # via connected_components_from_threshold (not just a bigger number
        # in isolation)?
        distances = self._distance_matrix()
        cluster_distances = distances.copy()
        np.fill_diagonal(cluster_distances, 0.0)

        threshold_k1 = MEMORY_POLICIES.estimate_cluster_threshold(distances, rarity_neighbors=1)
        _, clusters_k1 = MEMORY_POLICIES.connected_components_from_threshold(
            cluster_distances, threshold=threshold_k1
        )
        threshold_k4 = MEMORY_POLICIES.estimate_cluster_threshold(distances, rarity_neighbors=4)
        _, clusters_k4 = MEMORY_POLICIES.connected_components_from_threshold(
            cluster_distances, threshold=threshold_k4
        )
        self.assertLessEqual(len(clusters_k4), len(clusters_k1))

    def test_out_of_range_rarity_neighbors_does_not_crash(self):
        distances = self._distance_matrix()
        # More neighbors requested than points exist -- should clamp, not error.
        threshold = MEMORY_POLICIES.estimate_cluster_threshold(distances, rarity_neighbors=10_000)
        self.assertTrue(np.isfinite(threshold))


if __name__ == "__main__":
    unittest.main()
