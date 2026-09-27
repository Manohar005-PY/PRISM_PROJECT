"""Unit tests for ObservationExtractor bottom-center projection and diagnostic rejections."""

import unittest

from app.core.types import BoundingBox
from app.perception.model import (
    Detection,
    DetectionRejection,
    DetectionRejectionReason,
    PhysicalObservation,
)
from app.perception.observation import ObservationExtractor
from app.state.projection import (
    CameraIntrinsics,
    CameraPose,
    GroundPlaneProjector,
    ProjectionConfig,
)


class TestObservationExtractor(unittest.TestCase):
    """Test suite for ObservationExtractor."""

    def setUp(self) -> None:
        intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, image_width=640, image_height=480)
        pose = CameraPose(height=1.2, pitch=0.20)
        config = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
        self.projector = GroundPlaneProjector(config)
        self.extractor = ObservationExtractor(self.projector, min_confidence=0.30)

    def test_valid_observation_extraction(self) -> None:
        # Bbox in lower center of image (ground contact)
        bbox = BoundingBox(x_min=270.0, y_min=250.0, x_max=370.0, y_max=400.0)
        det = Detection("d1", 0, "person", 0.90, bbox, 0.0)

        result = self.extractor.extract_observation(det)
        self.assertIsInstance(result, PhysicalObservation)
        self.assertEqual(result.class_name, "person")  # type: ignore[union-attr]
        self.assertGreater(result.position.z, 0.5)  # type: ignore[union-attr]
        self.assertGreater(result.measurement_covariance[0][0], 0.0)  # type: ignore[union-attr]

    def test_low_confidence_rejection(self) -> None:
        bbox = BoundingBox(x_min=270.0, y_min=250.0, x_max=370.0, y_max=400.0)
        det = Detection("d1", 0, "person", 0.15, bbox, 0.0)  # Below 0.30 threshold

        result = self.extractor.extract_observation(det)
        self.assertIsInstance(result, DetectionRejection)
        self.assertEqual(result.reason, DetectionRejectionReason.LOW_CONFIDENCE)  # type: ignore[union-attr]

    def test_above_horizon_projection_rejection(self) -> None:
        # Bbox in top sky region of image (above horizon)
        bbox = BoundingBox(x_min=100.0, y_min=10.0, x_max=200.0, y_max=50.0)
        det = Detection("d1", 0, "person", 0.90, bbox, 0.0)

        result = self.extractor.extract_observation(det)
        self.assertIsInstance(result, DetectionRejection)
        self.assertEqual(result.reason, DetectionRejectionReason.ABOVE_HORIZON)  # type: ignore[union-attr]

    def test_out_of_bounds_rejection(self) -> None:
        # Bbox extending outside image boundaries
        bbox = BoundingBox(x_min=600.0, y_min=200.0, x_max=700.0, y_max=500.0)
        det = Detection("d1", 0, "person", 0.90, bbox, 0.0)

        result = self.extractor.extract_observation(det)
        self.assertIsInstance(result, DetectionRejection)
        self.assertEqual(result.reason, DetectionRejectionReason.OUT_OF_BOUNDS)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
