"""System tests evaluating graceful degradation and fail-safe behaviors under failure modes."""

import math
import unittest

from app.core.enums import RiskLevel
from app.core.types import BoundingBox, Vector2
from app.decision.engine import DecisionEngine
from app.decision.model import NavigationAction
from app.perception.adapters.synthetic import SyntheticDetector
from app.perception.model import Detection
from app.perception.pipeline import PerceptionPipeline
from app.prediction.constant_velocity import ConstantVelocityPredictor
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import RiskConfig
from app.runtime.config import RuntimeConfig
from app.runtime.pipeline import PRISMPipeline
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator
from app.state.projection import (
    CameraIntrinsics,
    CameraPose,
    GroundPlaneProjector,
    ProjectionConfig,
)
from app.tracking.config import TrackingConfig
from app.tracking.tracker import MultiObjectTracker


class TestFailureModes(unittest.TestCase):
    """Tests evaluating system fail-safe and error-recovery behaviors."""

    def setUp(self) -> None:
        intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
        pose = CameraPose(1.2, 0.15)
        proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
        self.projector = GroundPlaneProjector(proj_cfg)

        self.state_estimator = KalmanStateEstimator(KalmanFilterConfig())
        self.tracker = MultiObjectTracker(TrackingConfig(), state_estimator=self.state_estimator)
        self.detector = SyntheticDetector(config=self.projector.config, pixel_noise_sigma=0.5, random_seed=42)
        self.perc_pipe = PerceptionPipeline(self.detector, self.projector)

        self.pipeline = PRISMPipeline(
            perception_pipeline=self.perc_pipe,
            tracker=self.tracker,
            state_estimator=self.state_estimator,
            predictor=ConstantVelocityPredictor(),
            risk_evaluator=PredictiveRiskEvaluator(RiskConfig()),
            decision_engine=DecisionEngine(),
            config=RuntimeConfig(),
        )

    def test_empty_frame_handling(self) -> None:
        """Process empty frame without crashing."""
        res = self.pipeline.process_frame(raw_frame=None, timestamp=1.0, explicit_observations=[])
        self.assertEqual(res.detection_count, 0)
        self.assertEqual(res.decision.action, NavigationAction.CONTINUE)

    def test_out_of_order_timestamp_failsafe(self) -> None:
        """Process out-of-order timestamp triggering fail-safe mode."""
        self.pipeline.process_frame(raw_frame=None, timestamp=2.0, explicit_observations=[])
        # t=1.0 < t=2.0
        res = self.pipeline.process_frame(raw_frame=None, timestamp=1.0, explicit_observations=[])
        self.assertEqual(res.decision.action, NavigationAction.STOP)

    def test_above_horizon_detection_rejection(self) -> None:
        """Detection above horizon is safely rejected without numeric overflow."""
        v_horiz = self.projector.horizon_v
        det = Detection(
            detection_id="d_sky",
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=BoundingBox(300.0, v_horiz - 50.0, 340.0, v_horiz - 10.0),
            timestamp=1.0,
        )
        obs, rej = self.pipeline.perception_pipeline.extractor.extract_observations([det])
        self.assertEqual(len(obs), 0)
        self.assertEqual(len(rej), 1)


if __name__ == "__main__":
    unittest.main()
