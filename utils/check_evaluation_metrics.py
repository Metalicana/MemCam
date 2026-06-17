import importlib.util
from pathlib import Path

import numpy as np


def load_eval_module():
    module_path = Path(__file__).with_name("evaluate_context_memory.py")
    spec = importlib.util.spec_from_file_location("evaluate_context_memory", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    evaluator = load_eval_module()

    black = np.zeros((16, 16, 3), dtype=np.uint8)
    white = np.full((16, 16, 3), 255, dtype=np.uint8)
    half = np.full((16, 16, 3), 127, dtype=np.uint8)

    same = evaluator.frame_metrics(black, black)
    assert same["mae"] == 0.0
    assert same["mse"] == 0.0
    assert same["rmse"] == 0.0
    assert same["psnr_db"] == 100.0
    assert abs(same["ssim"] - 1.0) < 1e-9

    opposite = evaluator.frame_metrics(black, white)
    assert opposite["mae"] == 255.0
    assert opposite["mse"] == 255.0**2
    assert opposite["rmse"] == 255.0
    assert opposite["psnr_db"] == 0.0
    assert opposite["ssim"] < 0.01

    temporal = evaluator.temporal_delta_metrics(white, half, black, black)
    assert temporal["temporal_delta_mae"] == 128.0
    assert temporal["temporal_delta_rmse"] == 128.0

    print("evaluation metric checks passed")


if __name__ == "__main__":
    main()
