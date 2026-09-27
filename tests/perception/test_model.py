"""Unit tests for perception domain models (BoundingBox, Detection, PhysicalObservation, TrackedObservation)."""

import unittest

from app.core.types import BoundingBox, Vector2
from app.perception.model import (
    Detection,
    DetectionRejection,
    DetectionRejectionReason,
    PhysicalObservation,
    TrackedObservation,
)


class TestPerceptionModel(unittest.TestCase):
    """Test suite for perception domain models."""

    def test_bounding_box_properties(self) -> None:
        bbox = BoundingBox(x_min=100.0, y_min=200.0, x_max=300.0, y_max=500.0)
        self.assertEqual(bbox.width, 200.0)
        self.assertEqual(bbox.height, 300.0)
        self.assertEqual(bbox.center, (200.0, 350.0))
        self.assertEqual(bbox.bottom_center, (200.0, 500.0))

    def test_detection_creation_and_immutability(self) -> None:
        bbox = BoundingBox(10.0, 20.0, 30.0, 40.0)
        det = Detection(
            detection_id="det_1",
            class_id=0,
            class_name="person",
            confidence=0.92,
            bbox=bbox,
            timestamp=1.5,
        )
        self.assertEqual(det.detection_id, "det_1")
        self.assertEqual(det.class_name, "person")
        self.assertEqual(det.confidence, 0.92)

        with self.assertRaises(Exception):
            det.confidence = 0.5  # type: ignore[misc]

    def test_physical_observation_valid_covariance(self) -> None:
        cov = ((0.04, 0.0), (0.0, 0.09))
        obs = PhysicalObservation(
            position=Vector2(1.2, 3.5),
            measurement_covariance=cov,
            confidence=0.88,
            timestamp=2.0,
            detection_id="det_1",
            class_name="person",
        )
        self.assertEqual(obs.position, Vector2(1.2, 3.5))
        self.assertEqual(obs.measurement_covariance, cov)
        self.assertEqual(obs.confidence, 0.88)

    def test_tracked_observation_association(self) -> None:
        cov = ((0.04, 0.0), (0.0, 0.09))
        obs = PhysicalObservation(
            position=Vector2(1.2, 3.5),
            measurement_covariance=cov,
            confidence=0.88,
            timestamp=2.0,
            detection_id="det_1",
            class_name="person",
        )
        tracked = TrackedObservation(track_id=42, observation=obs)
        self.assertEqual(tracked.track_id, 42)
        self.assertEqual(tracked.observation.position, Vector2(1.2, 3.5))

    def test_physical_observation_invalid_covariance_raises(self) -> None:
        # Non-positive diagonal
        with self.assertRaises(ValueError):
            PhysicalObservation(
                position=Vector2(1.0, 2.0),
                measurement_covariance=((0.0, 0.0), (0.0, 0.05)),
                confidence=0.8,
                timestamp=1.0,
                detection_id="det_err",
            )

        # Asymmetric
        with self.assertRaises(ValueError):
            PhysicalObservation(
                position=Vector2(1.0, 2.0),
                measurement_covariance=((0.04, 0.05), (0.01, 0.09)),
                confidence=0.8,
                timestamp=1.0,
                detection_id="det_err",
            )


if __name__ == "__main__":
    unittest.main()
