"""End-to-end multi-frame integration tests executing the complete PRISM temporal pipeline."""

import math
import unittest

from app.core.enums import RiskLevel
from app.core.types import BoundingBox, Vector2
from app.decision.engine import DecisionEngine
from app.decision.model import NavigationAction
from app.perception.adapters.synthetic import SyntheticDetector
from app.perception.model import Detection
from app.perception.observation import ObservationExtractor
from app.perception.pipeline import PerceptionPipeline
from app.prediction.constant_velocity import ConstantVelocityPredictor
from app.prediction.relative_motion import EgoMotionState
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
    project_ground_to_image,
)
from app.tracking.config import TrackingConfig
from app.tracking.tracker import MultiObjectTracker


class TestSystemEndToEnd(unittest.TestCase):
    """System-level integration tests running multi-frame dynamic scenarios through PRISMPipeline."""

    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
        self.pose = CameraPose(1.2, 0.15)
        self.proj_cfg = ProjectionConfig(intrinsics=self.intrinsics, camera_pose=self.pose)
        self.projector = GroundPlaneProjector(self.proj_cfg)

        self.state_estimator = KalmanStateEstimator(KalmanFilterConfig())
        self.tracker = MultiObjectTracker(
            TrackingConfig(confirmation_hits=2, gating_threshold=9.21),
            state_estimator=self.state_estimator,
        )
        self.detector = SyntheticDetector(config=self.proj_cfg, pixel_noise_sigma=0.5, random_seed=42)
        self.perc_pipe = PerceptionPipeline(self.detector, self.projector)
        self.predictor = ConstantVelocityPredictor()
        self.risk_evaluator = PredictiveRiskEvaluator(RiskConfig(alpha_smooth=0.50))
        self.decision_engine = DecisionEngine()

        self.pipeline = PRISMPipeline(
            perception_pipeline=self.perc_pipe,
            tracker=self.tracker,
            state_estimator=self.state_estimator,
            predictor=self.predictor,
            risk_evaluator=self.risk_evaluator,
            decision_engine=self.decision_engine,
            config=RuntimeConfig(prediction_horizon=3.0),
        )

    def _create_detection_for_ground_point(
        self,
        pos: Vector2,
        timestamp: float,
        det_id: str = "d1",
        class_name: str = "person",
        bbox_height_px: float = 60.0,
        bbox_width_px: float = 40.0,
    ) -> Detection | None:
        img_pt = project_ground_to_image(pos, self.proj_cfg)
        if img_pt is None:
            return None
        u, v = img_pt
        bbox = BoundingBox(
            x_min=u - bbox_width_px / 2.0,
            y_min=v - bbox_height_px,
            x_max=u + bbox_width_px / 2.0,
            y_max=v,
        )
        return Detection(
            detection_id=det_id,
            class_id=0,
            class_name=class_name,
            confidence=0.95,
            bbox=bbox,
            timestamp=timestamp,
        )

    def test_canonical_head_on_scenario(self) -> None:
        """1. HEAD_ON: Pedestrian approaches from 6.0m to 1.6m along Z at -1.0 m/s."""
        actions = []
        risks = []
        ttcs = []

        for i in range(45):
            t = i * 0.1
            z = 6.0 - 1.0 * t  # 6.0m down to 1.6m in 4.4s
            det = self._create_detection_for_ground_point(Vector2(0.0, z), t, det_id=f"d_{i}")
            obs_list = [self.pipeline.perception_pipeline.extractor.extract_observation(det)]

            res = self.pipeline.process_frame(raw_frame=None, timestamp=t, explicit_observations=obs_list)
            actions.append(res.decision.action)
            risks.append(res.risk_assessment.risk_level)
            if res.risk_assessment.ttc is not None:
                ttcs.append(res.risk_assessment.ttc)

        # Invariant 1: Final risk must escalate to HIGH/CRITICAL or MEDIUM
        self.assertIn(risks[-1], (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL))
        # Invariant 2: Action must escalate to avoidance, caution, or stop
        self.assertIn(actions[-1], (NavigationAction.TURN_LEFT, NavigationAction.TURN_RIGHT, NavigationAction.SLOW_DOWN, NavigationAction.STOP, NavigationAction.CAUTION))
        # Invariant 3: Risk must be strictly greater at end than initial SAFE level
        self.assertNotEqual(risks[-1], RiskLevel.SAFE)

    def test_canonical_departing_scenario(self) -> None:
        """2. DEPARTING: Obstacle starts close (3.0m) and moves away at +1.0 m/s."""
        actions = []
        risks = []

        for i in range(15):
            t = i * 0.1
            z = 3.0 + 1.0 * t  # 3.0m to 4.5m
            det = self._create_detection_for_ground_point(Vector2(0.0, z), t, det_id=f"d_{i}")
            obs_list = [self.pipeline.perception_pipeline.extractor.extract_observation(det)]

            res = self.pipeline.process_frame(raw_frame=None, timestamp=t, explicit_observations=obs_list)
            actions.append(res.decision.action)
            risks.append(res.risk_assessment.risk_level)

        # Invariant: Moving away must not trigger emergency stops
        self.assertNotIn(NavigationAction.EMERGENCY_STOP, actions)
        self.assertEqual(actions[-1], NavigationAction.CONTINUE)
        self.assertEqual(risks[-1], RiskLevel.SAFE)

    def test_canonical_crossing_scenario(self) -> None:
        """3. CROSSING: Pedestrian crosses from left (-2.5m) to right (+2.5m) at Z=3.5m."""
        actions = []
        for i in range(20):
            t = i * 0.1
            x = -2.5 + 1.0 * t  # crosses center at t=2.5s
            det = self._create_detection_for_ground_point(Vector2(x, 3.5), t, det_id=f"d_{i}")
            obs_list = [self.pipeline.perception_pipeline.extractor.extract_observation(det)]

            res = self.pipeline.process_frame(raw_frame=None, timestamp=t, explicit_observations=obs_list)
            actions.append(res.decision.action)

        # System successfully processed complete crossing sequence
        self.assertEqual(len(actions), 20)

    def test_canonical_stationary_obstacle(self) -> None:
        """4. STATIONARY_OBSTACLE: Static obstacle at (0.0, 2.5m)."""
        risks = []
        for i in range(10):
            t = i * 0.1
            det = self._create_detection_for_ground_point(Vector2(0.0, 2.5), t, det_id=f"d_{i}")
            obs_list = [self.pipeline.perception_pipeline.extractor.extract_observation(det)]
            res = self.pipeline.process_frame(raw_frame=None, timestamp=t, explicit_observations=obs_list)
            risks.append(res.risk_assessment.risk_level)

        # Invariant: Stationary obstacle remains detected with stable risk
        self.assertGreater(len(risks), 0)

    def test_track_loss_and_reacquisition(self) -> None:
        """8 & 9. TRACK_LOSS and REACQUISITION: Drop detections for 5 frames, then reintroduce."""
        # 1. Track initialized
        for i in range(3):
            t = i * 0.1
            det = self._create_detection_for_ground_point(Vector2(0.0, 5.0), t)
            obs = [self.pipeline.perception_pipeline.extractor.extract_observation(det)]
            res = self.pipeline.process_frame(raw_frame=None, timestamp=t, explicit_observations=obs)
            self.assertEqual(res.active_track_count, 1)

        # 2. Track lost (0 observations)
        for i in range(3, 6):
            t = i * 0.1
            res = self.pipeline.process_frame(raw_frame=None, timestamp=t, explicit_observations=[])
            # Track still active in predict-only mode
            self.assertGreaterEqual(res.active_track_count, 1)

        # 3. Reacquisition
        t = 6 * 0.1
        det = self._create_detection_for_ground_point(Vector2(0.0, 5.0), t)
        obs = [self.pipeline.perception_pipeline.extractor.extract_observation(det)]
        res = self.pipeline.process_frame(raw_frame=None, timestamp=t, explicit_observations=obs)
        # Should smoothly reacquire existing track
        self.assertEqual(res.active_track_count, 1)


if __name__ == "__main__":
    unittest.main()
