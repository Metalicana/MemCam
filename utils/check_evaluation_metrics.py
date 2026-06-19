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

    fvd_runner = object.__new__(evaluator.FVDRunner)
    fvd_runner.eps = 1e-6
    features = np.arange(24, dtype=np.float64).reshape(6, 4)
    same_fvd = fvd_runner._frechet_distance(features, features.copy())
    assert abs(same_fvd) < 1e-8

    shifted_fvd = fvd_runner._frechet_distance(features, features + 1.0)
    assert shifted_fvd > 0.0

    rng = np.random.default_rng(0)
    rank_deficient = rng.normal(size=(6, 32))
    perturbed = rank_deficient + rng.normal(scale=0.1, size=rank_deficient.shape)
    assert abs(fvd_runner._frechet_distance(rank_deficient, rank_deficient.copy())) < 1e-8
    assert fvd_runner._frechet_distance(rank_deficient, perturbed) > 0.0

    print("evaluation metric checks passed")


if __name__ == "__main__":
    main()
