import importlib.util
from pathlib import Path


MEMORY_POLICIES_PATH = Path(__file__).resolve().parents[1] / "diffsynth" / "pipelines" / "memory_policies.py"
spec = importlib.util.spec_from_file_location("memory_policies", MEMORY_POLICIES_PATH)
memory_policies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory_policies)
FrameMemoryBuffer = memory_policies.FrameMemoryBuffer


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


if __name__ == "__main__":
    check_unbounded()
    check_fifo()
    check_fifo_requires_budget()
    print("memory policy checks passed")
