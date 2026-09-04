import unittest

import numpy as np

from utils.analyze_anchor_chain_consistency import (
    UE_FROM_CV,
    add_current_quality_aliases,
    build_pose_anchor_schedule,
    camera_intrinsics,
    correspondence_errors,
    relative_camera_motion,
    transform_points,
)


def yaw_c2w(degrees, translation=None):
    angle = np.deg2rad(float(degrees))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    if translation is not None:
        matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


class AnchorChainConsistencyTests(unittest.TestCase):
    def test_current_quality_aliases_support_shared_estimator_schema(self):
        rows = [
            {
                "current_psnr_db": 12.5,
                "current_ssim": 0.42,
                "anchor_psnr_db": 20.0,
                "anchor_ssim": 0.8,
            }
        ]
        add_current_quality_aliases(rows)
        self.assertEqual(rows[0]["psnr_db"], 12.5)
        self.assertEqual(rows[0]["ssim"], 0.42)

    def test_intrinsics_match_requested_fov(self):
        intrinsics = camera_intrinsics(640, 360, fov_half_h=45, fov_half_v=30)
        self.assertAlmostEqual(intrinsics[0, 0], 320.0)
        self.assertAlmostEqual(intrinsics[1, 1], 180.0 / np.tan(np.deg2rad(30)))
        self.assertEqual(intrinsics[0, 2], 320.0)
        self.assertEqual(intrinsics[1, 2], 180.0)

    def test_identical_ue_poses_have_identity_relative_cv_rotation(self):
        c2w = np.eye(4)
        rotation, translation = relative_camera_motion(c2w, c2w)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(translation, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(UE_FROM_CV.T @ UE_FROM_CV, np.eye(3))

    def test_rotation_correspondence_has_zero_transfer_error(self):
        anchor = yaw_c2w(0)
        current = yaw_c2w(12)
        intrinsics = camera_intrinsics(640, 360)
        rotation, _ = relative_camera_motion(anchor, current)
        first = np.asarray([[220.0, 140.0], [320.0, 180.0], [410.0, 205.0]])
        homography = intrinsics @ rotation @ np.linalg.inv(intrinsics)
        second = transform_points(homography, first)
        errors, mode = correspondence_errors(
            first, second, anchor, current, intrinsics
        )
        self.assertEqual(mode, "rotation")
        np.testing.assert_allclose(errors, np.zeros(len(first)), atol=1e-8)

    def test_translated_3d_points_satisfy_epipolar_constraint(self):
        anchor = yaw_c2w(0)
        current = yaw_c2w(5, translation=[0.0, 1.0, 0.0])
        intrinsics = camera_intrinsics(640, 360)
        r_anchor = anchor[:3, :3] @ UE_FROM_CV
        r_current = current[:3, :3] @ UE_FROM_CV
        world_points = np.asarray(
            [[8.0, -1.0, 0.2], [12.0, 2.0, -0.5], [18.0, 4.0, 1.0]]
        )

        def project(c2w, rotation, points):
            camera = (rotation.T @ (points - c2w[:3, 3]).T).T
            pixels = (intrinsics @ camera.T).T
            return pixels[:, :2] / pixels[:, 2:3]

        first = project(anchor, r_anchor, world_points)
        second = project(current, r_current, world_points)
        errors, mode = correspondence_errors(
            first, second, anchor, current, intrinsics
        )
        self.assertEqual(mode, "epipolar")
        np.testing.assert_allclose(errors, np.zeros(len(first)), atol=1e-8)

    def test_anchor_schedule_hands_off_before_overlap_disappears(self):
        c2ws = np.stack([yaw_c2w(angle) for angle in (0, 20, 40, 60, 80)])
        schedule = build_pose_anchor_schedule(
            c2ws,
            frame_stride=1,
            handoff_overlap=0.50,
            min_pair_overlap=0.10,
        )
        handoffs = [row for row in schedule if row["scheduled_handoff"]]
        self.assertTrue(handoffs)
        self.assertTrue(all(row["evaluable"] for row in handoffs))
        first_handoff = handoffs[0]
        self.assertEqual(first_handoff["anchor_frame"], 0)
        self.assertGreaterEqual(first_handoff["pose_overlap"], 0.10)


if __name__ == "__main__":
    unittest.main()
