"""Unit tests for SyntheticDetector deterministic forward generation."""

import unittest

from app.core.types import Vector2
from app.perception.adapters.synthetic import SyntheticDetector
from app.state.projection import CameraIntrinsics, CameraPose, ProjectionConfig


class TestSyntheticDetector(unittest.TestCase):
    """Test suite for SyntheticDetector."""

    def test_deterministic_seed_reproducibility(self) -> None:
        """Identical seeds must produce identical noisy bounding box detections."""
        intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, image_width=640, image_height=480)
        pose = CameraPose(height=1.2, pitch=0.15)
        config = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)

        d1 = SyntheticDetector(config=config, pixel_noise_sigma=3.0, random_seed=123)
        d2 = SyntheticDetector(config=config, pixel_noise_sigma=3.0, random_seed=123)

        objs = [(1, "person", Vector2(0.5, 3.0)), (2, "chair", Vector2(-1.0, 4.0))]

        dets1 = d1.detect(objs, timestamp=0.0)
        dets2 = d2.detect(objs, timestamp=0.0)

        self.assertEqual(len(dets1), 2)
        self.assertEqual(len(dets2), 2)

        for det1, det2 in zip(dets1, dets2):
            self.assertAlmostEqual(det1.bbox.x_min, det2.bbox.x_min)
            self.assertAlmostEqual(det1.bbox.y_max, det2.bbox.y_max)

    def test_forward_projection_positions(self) -> None:
        """Closer objects must produce larger bounding boxes than distant objects."""
        detector = SyntheticDetector(pixel_noise_sigma=0.0)
        objs = [(1, "person", Vector2(0.0, 2.0)), (2, "person", Vector2(0.0, 6.0))]
        dets = detector.detect(objs, timestamp=0.0)

        self.assertEqual(len(dets), 2)
        near_det = dets[0]
        far_det = dets[1]

        self.assertGreater(near_det.bbox.height, far_det.bbox.height)
        self.assertGreater(near_det.bbox.width, far_det.bbox.width)


if __name__ == "__main__":
    unittest.main()
