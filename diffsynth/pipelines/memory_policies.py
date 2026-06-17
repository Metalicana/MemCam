from collections import OrderedDict


SUPPORTED_MEMORY_POLICIES = ("unbounded", "fifo")


class FrameMemoryBuffer:
    def __init__(self, policy="unbounded", budget=None):
        if policy not in SUPPORTED_MEMORY_POLICIES:
            raise ValueError(
                f"Unsupported memory policy '{policy}'. "
                f"Expected one of {SUPPORTED_MEMORY_POLICIES}."
            )
        if policy == "fifo" and budget is None:
            raise ValueError("FIFO memory policy requires an explicit memory budget")
        if budget is not None and budget <= 0:
            raise ValueError("memory budget must be positive when provided")

        self.policy = policy
        self.budget = budget
        self._frames = OrderedDict()

    def add(self, frame_idx):
        evicted = []
        self._frames[frame_idx] = None
        if self.policy == "fifo" and self.budget is not None:
            while len(self._frames) > self.budget:
                evicted_frame_idx, _ = self._frames.popitem(last=False)
                evicted.append(evicted_frame_idx)
        return evicted

    def update(self, frame_indices):
        evicted = []
        for frame_idx in frame_indices:
            evicted.extend(self.add(frame_idx))
        return evicted

    def candidates(self, exclude_frames=None):
        exclude_frames = set(exclude_frames or [])
        return [frame_idx for frame_idx in self._frames.keys() if frame_idx not in exclude_frames]

    def __len__(self):
        return len(self._frames)
