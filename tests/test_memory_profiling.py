import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "diffsynth" / "pipelines" / "memory_profiling.py"
SPEC = importlib.util.spec_from_file_location("memory_profiling", MODULE_PATH)
PROFILING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILING)


class MemoryProfilingTest(unittest.TestCase):
    def test_storage_accounting(self):
        tensors = {
            0: torch.zeros(2, 3, dtype=torch.float32),
            1: torch.zeros(4, dtype=torch.bfloat16),
        }
        arrays = {
            0: np.zeros((2, 5), dtype=np.float32),
            1: np.zeros((3,), dtype=np.float64),
        }
        self.assertEqual(PROFILING.tensor_mapping_nbytes(tensors), 32)
        self.assertEqual(PROFILING.numpy_mapping_nbytes(arrays), 64)

    def test_cpu_profiler_writes_section_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "profile.jsonl"
            profiler = PROFILING.MemoryRolloutProfiler(
                path=path,
                metadata={"run_name": "test"},
                cuda_device="cpu",
            )
            profiler.start_rollout(stored_memory_size=1, bank_frame_bytes=12)
            profiler.begin_section(0)
            with profiler.phase("work"):
                pass
            profiler.end_section(
                section_end_frame=76,
                generated_seconds=10,
                stored_memory_size=16,
                candidate_count=1,
                bank_frame_bytes=192,
                bank_feature_bytes=8,
                host_to_device_bytes=12,
            )
            profiler.finish_rollout()
            text = path.read_text(encoding="utf-8")

        self.assertIn('"event": "section_profile"', text)
        self.assertIn('"event": "rollout_summary"', text)
        self.assertIn('"run_name": "test"', text)


if __name__ == "__main__":
    unittest.main()
