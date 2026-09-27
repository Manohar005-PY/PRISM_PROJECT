"""Unit tests for PerceptionPipeline orchestration and rejection handling."""

import unittest

from app.core.types import BoundingBox, Vector2
from app.perception.adapters.synthetic import SyntheticDetector
from app.perception.model import (
    Detection,
    DetectionRejectionReason,
    PhysicalObservation,
)
from app.perception.pipeline import PerceptionPipeline
from app.state.projection import (
    CameraIntrinsics,
    CameraPose,
    GroundPlaneProjector,
    ProjectionConfig,
)


class TestPerceptionPipeline(unittest.TestCase):
    """Test suite for PerceptionPipeline."""

    def setUp(self) -> None:
        intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, image_width=640, image_height=480)
        pose = CameraPose(height=1.2, pitch=0.20)
        config = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
        self.projector = GroundPlaneProjector(config)
        self.detector = SyntheticDetector(config=config)
        self.pipeline = PerceptionPipeline(self.detector, self.projector, min_confidence=0.20)

    def test_pipeline_valid_frame(self) -> None:
        objs = [(1, "person", Vector2(0.0, 3.0)), (2, "person", Vector2(1.5, 4.0))]
        obs, rejections = self.pipeline.process_frame(frame=objs, timestamp=0.0)

        self.assertEqual(len(obs), 2)
        self.assertEqual(len(rejections), 0)

        for o in obs:
            self.assertIsInstance(o, PhysicalObservation)
            self.assertGreater(o.position.z, 0.0)

    def test_pipeline_mixed_valid_and_rejected_detections(self) -> None:
        # Construct explicit detections: one valid, one low confidence, one above horizon
        valid_bbox = BoundingBox(270.0, 250.0, 370.0, 400.0)
        sky_bbox = BoundingBox(100.0, 10.0, 200.0, 50.0)

        det_valid = Detection("d1", 0, "person", 0.90, valid_bbox, 0.0)
        det_low_conf = Detection("d2", 0, "person", 0.10, valid_bbox, 0.0)
        det_sky = Detection("d3", 0, "person", 0.90, sky_bbox, 0.0)

        obs, rejections = self.pipeline.process_frame(frame=[det_valid, det_low_conf, det_sky], timestamp=0.0)

        self.assertEqual(len(obs), 1)
        self.assertEqual(len(rejections), 2)

        rejection_reasons = {r.reason for r in rejections}
        self.assertIn(DetectionRejectionReason.LOW_CONFIDENCE, rejection_reasons)
        self.assertIn(DetectionRejectionReason.ABOVE_HORIZON, rejection_reasons)


if __name__ == "__main__":
    unittest.main()
