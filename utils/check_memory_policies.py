import importlib.util
from pathlib import Path

import numpy as np


MEMORY_POLICIES_PATH = Path(__file__).resolve().parents[1] / "diffsynth" / "pipelines" / "memory_policies.py"
spec = importlib.util.spec_from_file_location("memory_policies", MEMORY_POLICIES_PATH)
memory_policies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory_policies)
FrameMemoryBuffer = memory_policies.FrameMemoryBuffer
compute_rarity_irreplaceability_scores = memory_policies.compute_rarity_irreplaceability_scores


def check_unbounded():
    memory = FrameMemoryBuffer(policy="unbounded")
    memory.update(range(5))
    assert memory.candidates() == [0, 1, 2, 3, 4]
    assert memory.candidates(exclude_frames={1, 3}) == [0, 2, 4]


def check_fifo():
    memory = FrameMemoryBuffer(policy="fifo", budget=3)
    evicted = memory.update(range(5))
    assert evicted == [0, 1]
    assert memory.candidates() == [2, 3, 4]
    evicted = memory.add(5)
    assert evicted == [2]
    assert memory.candidates() == [3, 4, 5]
    assert memory.candidates(exclude_frames={4}) == [3, 5]


def check_fifo_requires_budget():
    try:
        FrameMemoryBuffer(policy="fifo")
    except ValueError:
        return
    raise AssertionError("FIFO without a budget should fail")


def make_line_c2ws(num_frames):
    c2ws = np.repeat(np.eye(4, dtype=np.float64)[None], num_frames, axis=0)
    c2ws[:, 0, 3] = np.arange(num_frames, dtype=np.float64)
    return c2ws


def check_rarity_irreplaceability_budgeting():
    memory = FrameMemoryBuffer(
        policy="rarity_irreplaceability",
        budget=3,
        pinned_frames={0},
    )
    scores = {0: float("inf"), 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    evicted = memory.update(range(5), eviction_scores=scores, protected_frames={4})
    assert evicted == [1, 2]
    assert memory.candidates() == [0, 3, 4]


def check_rarity_irreplaceability_requires_budget():
    try:
        FrameMemoryBuffer(policy="rarity_irreplaceability")
    except ValueError:
        return
    raise AssertionError("rarity_irreplaceability without a budget should fail")


def check_rarity_irreplaceability_scores():
    memory = FrameMemoryBuffer(
        policy="rarity_irreplaceability",
        budget=4,
        pinned_frames={0},
    )
    memory.update([0, 1, 2, 5], eviction_scores={idx: 0.0 for idx in [0, 1, 2, 5]})
    memory.record_selection(2, 0.8)

    scores = compute_rarity_irreplaceability_scores(
        c2ws=make_line_c2ws(8),
        memory_frame_indices=memory.candidates(),
        memory_buffer=memory,
        future_frame_indices=[6, 7],
        pinned_frames={0},
    )
    assert set(scores) == {0, 1, 2, 5}
    assert scores[0] == float("inf")
    assert scores[5] > scores[1]


if __name__ == "__main__":
    check_unbounded()
    check_fifo()
    check_fifo_requires_budget()
    check_rarity_irreplaceability_budgeting()
    check_rarity_irreplaceability_requires_budget()
    check_rarity_irreplaceability_scores()
    print("memory policy checks passed")
