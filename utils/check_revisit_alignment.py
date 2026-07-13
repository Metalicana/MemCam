import math

import numpy as np
from PIL import Image

from calibrate_revisit_alignment import (
    align_input_to_output,
    build_pixel_to_ue_matrix,
    build_threshold_sweep,
    masked_gradient_rmse,
    masked_rgb_rmse,
    rotation_homography_output_to_input,
    select_best_thresholds,
)


def yaw_c2w(degrees):
    angle = math.radians(degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return c2w


def synthetic_image(width=160, height=96):
    y, x = np.mgrid[0:height, 0:width]
    array = np.stack(
        [
            255.0 * x / max(width - 1, 1),
            255.0 * y / max(height - 1, 1),
            127.5 + 100.0 * np.sin(x / 11.0) * np.cos(y / 13.0),
        ],
        axis=-1,
    )
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def direction_image(c2w, width=320, height=192):
    pixel_to_ue = build_pixel_to_ue_matrix(width, height, 90.0, 60.0)
    y, x = np.mgrid[0:height, 0:width]
    pixels = np.stack([x, y, np.ones_like(x)], axis=0).reshape(3, -1)
    camera_rays = pixel_to_ue @ pixels
    camera_rays /= np.linalg.norm(camera_rays, axis=0, keepdims=True)
    world_rays = c2w[:3, :3] @ camera_rays
    colors = 127.5 * (world_rays.T.reshape(height, width, 3) + 1.0)
    return Image.fromarray(np.clip(colors, 0, 255).astype(np.uint8))


def check_identity_alignment():
    image = synthetic_image()
    reference, aligned, mask = align_input_to_output(
        image,
        image,
        np.eye(4),
        np.eye(4),
        90.0,
        60.0,
    )
    assert np.mean(mask) > 0.90
    assert masked_rgb_rmse(reference, aligned, mask) == 0.0
    assert masked_gradient_rmse(reference, aligned, mask) == 0.0


def check_rotated_view_alignment():
    first = yaw_c2w(0.0)
    second = yaw_c2w(25.0)
    reference, aligned, mask = align_input_to_output(
        direction_image(first),
        direction_image(second),
        first,
        second,
        90.0,
        60.0,
    )
    assert 0.5 < np.mean(mask) < 0.9
    assert masked_rgb_rmse(reference, aligned, mask) < 0.005


def check_rotation_homographies_are_inverses():
    first = yaw_c2w(0.0)
    second = yaw_c2w(25.0)
    first_to_second = rotation_homography_output_to_input(
        first, second, 160, 96, 90.0, 60.0
    )
    second_to_first = rotation_homography_output_to_input(
        second, first, 160, 96, 90.0, 60.0
    )
    product = first_to_second @ second_to_first
    product = product / product[2, 2]
    assert np.allclose(product, np.eye(3), atol=1e-8)


def check_pixel_to_ue_center_ray():
    matrix = build_pixel_to_ue_matrix(161, 97, 90.0, 60.0)
    ray = matrix @ np.array([80.0, 48.0, 1.0])
    assert np.allclose(ray, [1.0, 0.0, 0.0], atol=1e-12)


def check_threshold_selection():
    pairs = [
        {
            "row": 0,
            "frame_i": 0,
            "frame_j": 100,
            "position_distance": 0.1,
            "rotation_deg": 3.0,
            "overlap_fraction": 0.8,
            "rgb_rmse": 0.01,
        },
        {
            "row": 1,
            "frame_i": 0,
            "frame_j": 100,
            "position_distance": 0.4,
            "rotation_deg": 20.0,
            "overlap_fraction": 0.6,
            "rgb_rmse": 0.04,
        },
    ]
    sweep = build_threshold_sweep(
        pairs,
        total_videos=2,
        position_thresholds=[0.25, 0.5],
        rotation_thresholds=[5.0, 30.0],
        overlap_thresholds=[0.5],
        metric_fields=["rgb_rmse"],
    )
    best = select_best_thresholds(sweep)
    threshold_005 = [
        row for row in best if float(row["oracle_error_threshold"]) == 0.05
    ][0]
    assert threshold_005["videos_after_oracle_filter"] == 2
    assert threshold_005["position_threshold"] == 0.5
    assert threshold_005["rotation_threshold_deg"] == 30.0


def main():
    check_identity_alignment()
    check_rotated_view_alignment()
    check_rotation_homographies_are_inverses()
    check_pixel_to_ue_center_ray()
    check_threshold_selection()
    print("Revisit alignment checks passed.")


if __name__ == "__main__":
    main()
