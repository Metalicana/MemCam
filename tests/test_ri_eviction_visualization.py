import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "utils" / "visualize_ri_evictions.py"
SPEC = importlib.util.spec_from_file_location("ri_eviction_visualization", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def eviction(section, frame, score, cluster_size=2, nearest=0):
    return {
        "event": "memory_eviction",
        "section_idx": section,
        "section_end_frame": section * 4 + 4,
        "evicted_memory_frame": frame,
        "memory_policy": "rarity_irreplaceability",
        "eviction_score": score,
        "eviction_rarity": 0.5,
        "eviction_irreplaceability": score / 0.5,
        "eviction_cluster_size": cluster_size,
        "eviction_rgb_nearest_frame": nearest,
        "eviction_rgb_nearest_distance": score / 0.5,
    }


class RiEvictionVisualizationTest(unittest.TestCase):
    def test_reconstructs_bank_from_actual_evictions(self):
        events = [
            eviction(0, 1, 0.1),
            eviction(0, 2, 0.2),
            eviction(1, 3, 0.3),
            eviction(1, 5, 0.4),
            eviction(1, 6, 0.5),
            eviction(1, 7, 0.6),
        ]

        snapshots = MODULE.reconstruct_ri_snapshots(
            events,
            budget=3,
            frames_per_section=5,
        )

        self.assertEqual(snapshots[0]["prospective"], [0, 1, 2, 3, 4])
        self.assertEqual(snapshots[0]["retained"], [0, 3, 4])
        self.assertEqual(snapshots[1]["prospective"], [0, 3, 4, 5, 6, 7, 8])
        self.assertEqual(snapshots[1]["retained"], [0, 4, 8])

    def test_ignores_other_policy_evictions(self):
        events = [eviction(0, 1, 0.1)]
        other = dict(eviction(0, 2, 0.2), memory_policy="fifo")
        events.append(other)

        grouped = MODULE.ri_evictions_by_section(events)

        self.assertEqual([row["evicted_memory_frame"] for row in grouped[0]], [1])

    def test_duplicate_examples_use_large_clusters_and_distinct_sections(self):
        events = [
            eviction(0, 1, 0.04, cluster_size=3),
            eviction(0, 2, 0.03, cluster_size=5),
            eviction(1, 5, 0.02, cluster_size=4),
        ]

        rows = MODULE.select_duplicate_evictions(events, limit=2)

        self.assertEqual([row["evicted_memory_frame"] for row in rows], [2, 5])

    def test_status_distinguishes_old_and_new_evictions(self):
        snapshot = {
            "incoming": [5, 6],
            "evicted": [2, 5],
        }

        self.assertEqual(MODULE.frame_status(2, snapshot), "evicted_old")
        self.assertEqual(MODULE.frame_status(5, snapshot), "evicted_new")
        self.assertEqual(MODULE.frame_status(6, snapshot), "retained_new")


if __name__ == "__main__":
    unittest.main()
