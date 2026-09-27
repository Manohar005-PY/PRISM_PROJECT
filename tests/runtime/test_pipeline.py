"""Unit tests for PRISMPipeline synchronous execution."""

import unittest

from app.core.types import Vector2
from app.decision.engine import DecisionEngine
from app.decision.model import NavigationAction
from app.perception.adapters.synthetic import SyntheticDetector
from app.perception.pipeline import PerceptionPipeline
from app.prediction.constant_velocity import ConstantVelocityPredictor
from app.prediction.relative_motion import EgoMotionState
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import RiskConfig
from app.runtime.config import RuntimeConfig
from app.runtime.health import RuntimeStatus
from app.runtime.pipeline import PRISMPipeline
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator
from app.state.projection import CameraIntrinsics, CameraPose, GroundPlaneProjector, ProjectionConfig
from app.tracking.config import TrackingConfig
from app.tracking.tracker import MultiObjectTracker


class TestPRISMPipeline(unittest.TestCase):
    """Test suite for PRISMPipeline synchronous frame processing."""

    def setUp(self) -> None:
        intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
        pose = CameraPose(1.2, 0.15)
        proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
        projector = GroundPlaneProjector(proj_cfg)
        detector = SyntheticDetector(config=proj_cfg, pixel_noise_sigma=1.0, random_seed=42)
        perc_pipe = PerceptionPipeline(detector, projector)

        state_estimator = KalmanStateEstimator(KalmanFilterConfig())
        tracker = MultiObjectTracker(TrackingConfig(confirmation_hits=2), state_estimator=state_estimator)

        self.pipeline = PRISMPipeline(
            perception_pipeline=perc_pipe,
            tracker=tracker,
            state_estimator=state_estimator,
            predictor=ConstantVelocityPredictor(),
            risk_evaluator=PredictiveRiskEvaluator(RiskConfig()),
            decision_engine=DecisionEngine(),
            config=RuntimeConfig(),
        )

    def test_empty_frame_processing(self) -> None:
        """Verify pipeline handles empty frame without crashing and returns safe CONTINUE decision."""
        result = self.pipeline.process_frame([], timestamp=0.1)
        self.assertEqual(result.frame_id, 1)
        self.assertEqual(result.decision.action, NavigationAction.CONTINUE)
        self.assertEqual(result.active_track_count, 0)
        self.assertEqual(result.runtime_status, RuntimeStatus.HEALTHY)

    def test_single_obstacle_frame_processing(self) -> None:
        """Verify pipeline processes an incoming stationary obstacle."""
        sim_entities = [(1, "person", Vector2(0.0, 3.5))]
        result = self.pipeline.process_frame(sim_entities, timestamp=0.1)
        self.assertEqual(result.frame_id, 1)
        self.assertEqual(result.active_track_count, 1)
        self.assertGreater(result.processing_duration_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
