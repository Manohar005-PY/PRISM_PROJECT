"""Unit tests for failure isolation and fail-safe policy enforcement."""

import unittest

from app.core.types import Vector2
from app.decision.model import NavigationAction
from app.perception.adapters.synthetic import SyntheticDetector
from app.perception.pipeline import PerceptionPipeline
from app.runtime.config import RuntimeConfig
from app.runtime.health import RuntimeStatus
from app.runtime.pipeline import PRISMPipeline
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator
from app.state.projection import CameraIntrinsics, CameraPose, GroundPlaneProjector, ProjectionConfig
from app.tracking.config import TrackingConfig
from app.tracking.tracker import MultiObjectTracker


class FailingPerceptionPipeline:
    """Mock perception pipeline that deliberately raises exceptions."""

    def process_frame(self, raw_frame, timestamp):
        raise RuntimeError("Simulated camera hardware drop")


class TestRuntimeErrorHandling(unittest.TestCase):
    """Test suite for runtime error isolation and fail-safe recovery."""

    def test_perception_failure_isolation(self) -> None:
        """Verify pipeline handles perception hardware crashes gracefully."""
        cfg = RuntimeConfig(fail_safe_action=NavigationAction.STOP)
        state_estimator = KalmanStateEstimator(KalmanFilterConfig())
        tracker = MultiObjectTracker(TrackingConfig(), state_estimator=state_estimator)

        pipeline = PRISMPipeline(
            perception_pipeline=FailingPerceptionPipeline(),  # type: ignore[arg-type]
            tracker=tracker,
            state_estimator=state_estimator,
            config=cfg,
        )

        result = pipeline.process_frame(None, timestamp=0.1)
        self.assertEqual(result.frame_id, 1)
        self.assertEqual(result.observation_count, 0)
        self.assertIsNotNone(result.decision)

    def test_out_of_order_timestamp_fail_safe(self) -> None:
        """Verify out-of-order timestamps trigger fail-safe STOP."""
        intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
        pose = CameraPose(1.2, 0.15)
        proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
        projector = GroundPlaneProjector(proj_cfg)
        detector = SyntheticDetector(config=proj_cfg, pixel_noise_sigma=1.0, random_seed=42)
        perc_pipe = PerceptionPipeline(detector, projector)

        pipeline = PRISMPipeline(
            perception_pipeline=perc_pipe,
            config=RuntimeConfig(fail_safe_action=NavigationAction.STOP),
        )

        # Frame 1 at t=1.0
        pipeline.process_frame([], timestamp=1.0)
        # Frame 2 at t=0.5 (out of order!)
        res = pipeline.process_frame([], timestamp=0.5)

        self.assertEqual(res.decision.action, NavigationAction.STOP)


if __name__ == "__main__":
    unittest.main()
