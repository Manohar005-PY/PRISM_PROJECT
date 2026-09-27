"""Unit, round-trip, horizon, and scientific validation tests for GroundPlaneProjector."""

import math
import unittest

from app.core.interfaces import GroundProjector
from app.core.types import BoundingBox, Vector2
from app.state.projection import (
    CameraIntrinsics,
    CameraPose,
    GroundPlaneProjector,
    ProjectionConfig,
    project_ground_to_image,
)


class TestGroundPlaneProjector(unittest.TestCase):
    """Comprehensive test suite for camera calibration and ground-plane projection."""

    def setUp(self) -> None:
        # Standard synthetic calibration: 640x480, 1.5m height, 15 degree downward pitch (~0.2618 rad)
        self.intrinsics = CameraIntrinsics(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
        )
        self.pose = CameraPose(
            height=1.5,
            pitch=math.radians(15.0),  # 15 degrees downward
        )
        self.config = ProjectionConfig(
            intrinsics=self.intrinsics,
            camera_pose=self.pose,
            epsilon=1e-6,
            max_ground_distance=50.0,
        )
        self.projector = GroundPlaneProjector(self.config)

    def test_protocol_conformance(self) -> None:
        """Verify GroundPlaneProjector implements the GroundProjector protocol."""
        self.assertIsInstance(self.projector, GroundProjector)

    def test_intrinsics_and_pose_validation(self) -> None:
        """Invalid calibration and pose parameters must raise descriptive exceptions."""
        with self.assertRaises(ValueError):
            CameraIntrinsics(image_width=0, image_height=480, fx=500, fy=500, cx=320, cy=240)
        with self.assertRaises(ValueError):
            CameraIntrinsics(image_width=640, image_height=480, fx=-500, fy=500, cx=320, cy=240)
        with self.assertRaises(ValueError):
            CameraPose(height=0.0, pitch=0.2)
        with self.assertRaises(ValueError):
            CameraPose(height=-1.5, pitch=0.2)
        with self.assertRaises(ValueError):
            ProjectionConfig(intrinsics=self.intrinsics, camera_pose=self.pose, epsilon=-1e-5)
        with self.assertRaises(ValueError):
            ProjectionConfig(intrinsics=self.intrinsics, camera_pose=self.pose, max_ground_distance=-10.0)

    def test_closed_loop_numerical_round_trip(self) -> None:
        """NUMERICAL PRECISION TEST: Forward project (X, Z) -> (u, v) and invert (u, v) -> (X_est, Z_est).

        Verifies mathematical formulation consistency and sub-millimeter floating-point precision.
        """
        test_points = [
            Vector2(0.0, 2.0),
            Vector2(-1.5, 3.0),
            Vector2(1.5, 3.0),
            Vector2(0.0, 5.0),
            Vector2(-2.0, 5.0),
            Vector2(2.0, 5.0),
            Vector2(-3.0, 10.0),
            Vector2(0.0, 10.0),
            Vector2(3.0, 10.0),
            Vector2(-4.0, 20.0),
            Vector2(0.0, 20.0),
            Vector2(4.0, 20.0),
        ]

        for pt in test_points:
            with self.subTest(ground_point=pt):
                # 1. Forward projection
                img_coords = project_ground_to_image(pt, self.config)
                self.assertIsNotNone(img_coords, f"Forward projection failed for {pt}")
                u, v = img_coords  # type: ignore[misc]

                # 2. Inverse projection
                recovered = self.projector.project((u, v))
                self.assertIsNotNone(recovered, f"Inverse projection failed for pixel ({u}, {v})")

                # 3. Numerical error check (< 0.1 mm precision)
                error = (recovered - pt).magnitude  # type: ignore[operator]
                self.assertLess(error, 1e-4, f"Round-trip error {error}m exceeds tolerance for {pt}")

    def test_lateral_symmetry(self) -> None:
        """Verify symmetric points (-X, Z) and (+X, Z) produce symmetric image pixels around cx."""
        pt_left = Vector2(-2.5, 6.0)
        pt_right = Vector2(2.5, 6.0)

        img_left = project_ground_to_image(pt_left, self.config)
        img_right = project_ground_to_image(pt_right, self.config)
        self.assertIsNotNone(img_left)
        self.assertIsNotNone(img_right)

        u_left, v_left = img_left  # type: ignore[misc]
        u_right, v_right = img_right  # type: ignore[misc]

        # Vertical coordinate must be identical
        self.assertAlmostEqual(v_left, v_right, delta=1e-5)

        # Horizontal coordinates must be equidistant from cx
        cx = self.intrinsics.cx
        self.assertAlmostEqual(cx - u_left, u_right - cx, delta=1e-5)

        # Inversion recovers exact coordinates
        rec_left = self.projector.project((u_left, v_left))
        rec_right = self.projector.project((u_right, v_right))
        self.assertIsNotNone(rec_left)
        self.assertIsNotNone(rec_right)
        self.assertAlmostEqual(rec_left.x, -2.5, delta=1e-4)  # type: ignore[union-attr]
        self.assertAlmostEqual(rec_right.x, 2.5, delta=1e-4)  # type: ignore[union-attr]

    def test_depth_monotonicity(self) -> None:
        """Verify increasing depth Z1 < Z2 < Z3 causes image v to move toward horizon monotonically."""
        depths = [2.0, 4.0, 7.0, 12.0, 25.0]
        v_coords = []
        recovered_depths = []

        for z in depths:
            pt = Vector2(0.0, z)
            img = project_ground_to_image(pt, self.config)
            self.assertIsNotNone(img)
            u, v = img  # type: ignore[misc]
            v_coords.append(v)

            rec = self.projector.project((u, v))
            self.assertIsNotNone(rec)
            recovered_depths.append(rec.z)  # type: ignore[union-attr]

        # In image space, points farther away appear higher up (smaller v) toward horizon
        for i in range(len(v_coords) - 1):
            self.assertGreater(v_coords[i], v_coords[i + 1])
            self.assertLess(recovered_depths[i], recovered_depths[i + 1])

    def test_multiple_camera_pitches(self) -> None:
        """Round-trip consistency must hold across shallow, moderate, and steep camera pitches."""
        pitches = [
            math.radians(5.0),   # Shallow downward pitch (5 deg)
            math.radians(15.0),  # Moderate downward pitch (15 deg)
            math.radians(35.0),  # Steep downward pitch (35 deg)
        ]

        for pitch in pitches:
            pose = CameraPose(height=1.4, pitch=pitch)
            config = ProjectionConfig(intrinsics=self.intrinsics, camera_pose=pose)
            projector = GroundPlaneProjector(config)

            pt = Vector2(1.0, 5.0)
            img = project_ground_to_image(pt, config)
            self.assertIsNotNone(img, f"Failed forward project at pitch {math.degrees(pitch)} deg")
            rec = projector.project(img)  # type: ignore[arg-type]
            self.assertIsNotNone(rec)
            err = (rec - pt).magnitude  # type: ignore[operator]
            self.assertLess(err, 1e-4)

    def test_horizon_safeguards_and_rejection(self) -> None:
        """Pixels at or above the horizon line must return None (safe rejection, no explosion)."""
        v_horiz = self.projector.horizon_v
        cx = self.intrinsics.cx

        # Exactly at horizon
        self.assertIsNone(self.projector.project((cx, v_horiz)))

        # Above horizon (sky)
        self.assertIsNone(self.projector.project((cx, v_horiz - 10.0)))
        self.assertIsNone(self.projector.project((cx, 0.0)))

        # Points just below horizon but exceeding max_ground_distance
        self.assertIsNone(self.projector.project((cx, v_horiz + 0.01)))

    def test_maximum_distance_boundary(self) -> None:
        """Projected points exceeding max_ground_distance return None rather than being clamped."""
        config_max = ProjectionConfig(
            intrinsics=self.intrinsics,
            camera_pose=self.pose,
            max_ground_distance=15.0,
        )
        projector_max = GroundPlaneProjector(config_max)

        # 10m is within threshold
        img_10m = project_ground_to_image(Vector2(0.0, 10.0), config_max)
        self.assertIsNotNone(img_10m)
        self.assertIsNotNone(projector_max.project(img_10m))  # type: ignore[arg-type]

        # 20m exceeds 15m threshold -> must return None
        img_20m = project_ground_to_image(Vector2(0.0, 20.0), config_max)
        self.assertIsNotNone(img_20m)
        self.assertIsNone(projector_max.project(img_20m))  # type: ignore[arg-type]

    def test_bounding_box_integration(self) -> None:
        """Verify project_bbox uses bottom_center without discrepancy."""
        bbox = BoundingBox(x_min=280.0, y_min=180.0, x_max=360.0, y_max=320.0)
        self.assertEqual(bbox.bottom_center, (320.0, 320.0))

        res_from_bbox = self.projector.project_bbox(bbox)
        res_from_point = self.projector.project(bbox.bottom_center)

        self.assertIsNotNone(res_from_bbox)
        self.assertEqual(res_from_bbox, res_from_point)

    def test_image_noise_propagation_experiment(self) -> None:
        """SCIENTIFIC TEST: Measure how image-space pixel noise propagates to ground depth uncertainty at varying distances.

        Demonstrates that constant pixel localization error (e.g. +/- 2px) produces non-linear
        quadratic growth in physical depth error at larger distances, motivating Kalman measurement noise scaling.
        """
        # Ground points at near (3m) and far (10m) distances
        pt_near = Vector2(0.0, 3.0)
        pt_far = Vector2(0.0, 10.0)

        img_near = project_ground_to_image(pt_near, self.config)
        img_far = project_ground_to_image(pt_far, self.config)
        self.assertIsNotNone(img_near)
        self.assertIsNotNone(img_far)

        u_near, v_near = img_near  # type: ignore[misc]
        u_far, v_far = img_far  # type: ignore[misc]

        # Inject 2 pixel vertical perturbation
        delta_v = 2.0
        rec_near_perturbed = self.projector.project((u_near, v_near - delta_v))
        rec_far_perturbed = self.projector.project((u_far, v_far - delta_v))
        self.assertIsNotNone(rec_near_perturbed)
        self.assertIsNotNone(rec_far_perturbed)

        err_near_z = abs(rec_near_perturbed.z - pt_near.z)  # type: ignore[union-attr]
        err_far_z = abs(rec_far_perturbed.z - pt_far.z)  # type: ignore[union-attr]

        # Depth uncertainty at 10m is substantially larger than at 3m
        self.assertGreater(err_far_z, err_near_z * 4.0)

    def test_invalid_input_handling(self) -> None:
        """NaN/Inf coordinates return None, and invalid types raise TypeError."""
        self.assertIsNone(self.projector.project((float("nan"), 300.0)))
        self.assertIsNone(self.projector.project((320.0, float("inf"))))
        with self.assertRaises(TypeError):
            self.projector.project("invalid")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.projector.project((320.0,))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
