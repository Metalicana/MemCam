import importlib.util
from pathlib import Path


def load_analysis_module():
    module_path = Path(__file__).with_name("analyze_memory_policies.py")
    spec = importlib.util.spec_from_file_location("analyze_memory_policies", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    analysis = load_analysis_module()
    item = {
        "_row": 0,
        "scene": "synthetic",
        "start_frame": 0,
        "duration_sec": 10,
        "num_frames": 153,
    }
    overlap_map = {frame_idx: set() for frame_idx in range(153)}
    for target_frame in range(77, 153):
        overlap_map[target_frame] = {0, 10, 76}

    fifo_summary, _ = analysis.simulate_row(
        item=item,
        policy="fifo",
        budget=2,
        overlap_map=overlap_map,
    )
    belady_summary, _ = analysis.simulate_row(
        item=item,
        policy="belady",
        budget=2,
        overlap_map=overlap_map,
    )

    assert belady_summary["coverage"] == 1.0
    assert belady_summary["coverage"] > fifo_summary["coverage"]
    assert belady_summary["oracle_recall"] >= fifo_summary["oracle_recall"]

    print("memory policy analysis checks passed")


if __name__ == "__main__":
    main()
