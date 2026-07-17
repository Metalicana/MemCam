import contextlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is optional outside profiling runs.
    psutil = None


GIB = 1024 ** 3


def tensor_nbytes(tensor):
    if tensor is None:
        return 0
    return int(tensor.numel()) * int(tensor.element_size())


def tensor_mapping_nbytes(tensors):
    return sum(tensor_nbytes(tensor) for tensor in tensors.values())


def numpy_mapping_nbytes(arrays):
    total = 0
    for value in arrays.values():
        if isinstance(value, np.ndarray):
            total += int(value.nbytes)
    return total


class MemoryRolloutProfiler:
    """Write synchronized rollout latency and memory samples as JSONL."""

    def __init__(self, path=None, metadata=None, cuda_device="cuda"):
        self.path = Path(path) if path is not None else None
        self.metadata = dict(metadata or {})
        self.cuda_device = cuda_device
        self.enabled = self.path is not None
        self._handle = None
        self._rollout_start = None
        self._section_start = None
        self._section_idx = None
        self._section_phases = {}
        self._peak_rss_bytes = 0
        self._peak_bank_frame_bytes = 0
        self._last_record = None

        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w", encoding="utf-8")

    @property
    def cuda_enabled(self):
        return torch.cuda.is_available() and torch.device(self.cuda_device).type == "cuda"

    def synchronize(self):
        if self.enabled and self.cuda_enabled:
            torch.cuda.synchronize(self.cuda_device)

    def _rss_bytes(self):
        if psutil is None:
            return None
        return int(psutil.Process(os.getpid()).memory_info().rss)

    def _memory_sample(self):
        rss_bytes = self._rss_bytes()
        if rss_bytes is not None:
            self._peak_rss_bytes = max(self._peak_rss_bytes, rss_bytes)

        sample = {
            "rss_gb": None if rss_bytes is None else rss_bytes / GIB,
            "peak_rss_gb": None if rss_bytes is None else self._peak_rss_bytes / GIB,
            "cuda_allocated_gb": None,
            "cuda_reserved_gb": None,
            "peak_cuda_allocated_gb": None,
            "peak_cuda_reserved_gb": None,
            "device_used_gb": None,
        }
        if not self.cuda_enabled:
            return sample

        allocated = torch.cuda.memory_allocated(self.cuda_device)
        reserved = torch.cuda.memory_reserved(self.cuda_device)
        peak_allocated = torch.cuda.max_memory_allocated(self.cuda_device)
        peak_reserved = torch.cuda.max_memory_reserved(self.cuda_device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.cuda_device)
        sample.update(
            {
                "cuda_allocated_gb": allocated / GIB,
                "cuda_reserved_gb": reserved / GIB,
                "peak_cuda_allocated_gb": peak_allocated / GIB,
                "peak_cuda_reserved_gb": peak_reserved / GIB,
                "device_used_gb": (total_bytes - free_bytes) / GIB,
            }
        )
        return sample

    def _write(self, payload):
        if not self.enabled:
            return
        record = {**self.metadata, **payload}
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def start_rollout(self, stored_memory_size, bank_frame_bytes, bank_feature_bytes=0):
        if not self.enabled:
            return
        self.synchronize()
        if self.cuda_enabled:
            torch.cuda.reset_peak_memory_stats(self.cuda_device)
        self._rollout_start = time.perf_counter()
        self._peak_bank_frame_bytes = int(bank_frame_bytes)
        self._write(
            {
                "event": "rollout_start",
                "stored_memory_size": int(stored_memory_size),
                "bank_frame_bytes": int(bank_frame_bytes),
                "bank_feature_bytes": int(bank_feature_bytes),
                **self._memory_sample(),
            }
        )

    def begin_section(self, section_idx):
        if not self.enabled:
            return
        self.synchronize()
        self._section_idx = int(section_idx)
        self._section_start = time.perf_counter()
        self._section_phases = {}

    @contextlib.contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return

        self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            elapsed = time.perf_counter() - start
            self._section_phases[name] = self._section_phases.get(name, 0.0) + elapsed

    def start_phase(self):
        if not self.enabled:
            return None
        self.synchronize()
        return time.perf_counter()

    def end_phase(self, name, start):
        if not self.enabled or start is None:
            return
        self.synchronize()
        elapsed = time.perf_counter() - start
        self._section_phases[name] = self._section_phases.get(name, 0.0) + elapsed

    def end_section(
        self,
        *,
        section_end_frame,
        generated_seconds,
        stored_memory_size,
        candidate_count,
        bank_frame_bytes,
        bank_feature_bytes,
        host_to_device_bytes,
    ):
        if not self.enabled:
            return
        self.synchronize()
        now = time.perf_counter()
        section_latency = now - self._section_start
        cumulative_latency = now - self._rollout_start
        self._peak_bank_frame_bytes = max(self._peak_bank_frame_bytes, int(bank_frame_bytes))
        record = {
            "event": "section_profile",
            "section_idx": self._section_idx,
            "section_end_frame": int(section_end_frame),
            "generated_seconds": float(generated_seconds),
            "section_latency_s": float(section_latency),
            "cumulative_rollout_latency_s": float(cumulative_latency),
            "latency_per_generated_second": (
                float(cumulative_latency / generated_seconds) if generated_seconds > 0 else None
            ),
            "phase_latency_s": dict(self._section_phases),
            "stored_memory_size": int(stored_memory_size),
            "candidate_count": int(candidate_count),
            "bank_frame_bytes": int(bank_frame_bytes),
            "bank_frame_gb": int(bank_frame_bytes) / GIB,
            "bank_feature_bytes": int(bank_feature_bytes),
            "bank_feature_gb": int(bank_feature_bytes) / GIB,
            "host_to_device_bytes": int(host_to_device_bytes),
            "host_to_device_gb": int(host_to_device_bytes) / GIB,
            **self._memory_sample(),
        }
        self._last_record = record
        self._write(record)

    def finish_rollout(self):
        if not self.enabled:
            return None
        self.synchronize()
        total_latency = time.perf_counter() - self._rollout_start
        last_record = self._last_record or {}
        generated_seconds = last_record.get("generated_seconds")
        summary = {
            "event": "rollout_summary",
            "completed": True,
            "rollout_latency_s": float(total_latency),
            "generated_seconds": generated_seconds,
            "latency_per_generated_second": (
                float(total_latency / generated_seconds) if generated_seconds else None
            ),
            "sections": None if self._section_idx is None else self._section_idx + 1,
            "stored_memory_size": last_record.get("stored_memory_size"),
            "peak_bank_frame_bytes": self._peak_bank_frame_bytes,
            "peak_bank_frame_gb": self._peak_bank_frame_bytes / GIB,
            **self._memory_sample(),
        }
        self._write(summary)
        self.close()
        return summary

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        self.close()
