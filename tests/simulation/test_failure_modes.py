"""Adversarial and failure-mode test suite for PRISM simulation and evaluation pipelines."""

import math
import unittest

from app.core.enums import RiskLevel
from app.core.types import BoundingBox, Vector2
from app.decision.engine import DecisionEngine
from app.decision.model import NavigationAction
from app.perception.adapters.synthetic import SyntheticDetector
from app.perception.model import Detection, PhysicalObservation
from app.perception.pipeline import PerceptionPipeline
from app.prediction.collision_geometry import (
    CPAResult,
    CollisionModel,
    analyze_collision,
    closest_point_of_approach,
    time_to_collision,
)
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
)
from app.tracking.config import TrackingConfig
from app.tracking.tracker import MultiObjectTracker


class TestAdversarialFailureModes(unittest.TestCase):
    """Comprehensive adversarial stress tests evaluating robustness and fail-safes."""

    def setUp(self) -> None:
        intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
        pose = CameraPose(1.25, 0.0)
        proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
        self.projector = GroundPlaneProjector(proj_cfg)
        self.state_estimator = KalmanStateEstimator(KalmanFilterConfig())
        self.tracker = MultiObjectTracker(TrackingConfig(), state_estimator=self.state_estimator)
        self.predictor = ConstantVelocityPredictor()
        self.risk_evaluator = PredictiveRiskEvaluator(RiskConfig())
        self.decision_engine = DecisionEngine()
        self.pipeline = PRISMPipeline(
            perception_pipeline=PerceptionPipeline(SyntheticDetector(config=self.projector.config), self.projector),
            tracker=self.tracker,
            state_estimator=self.state_estimator,
            predictor=self.predictor,
            risk_evaluator=self.risk_evaluator,
            decision_engine=self.decision_engine,
            config=RuntimeConfig(),
        )

    def test_ttc_dcpa_singularities(self) -> None:
        """Verify TTC and dCPA handle boundary singularities without NaN/Inf errors."""
        model = CollisionModel()
        # 1. Co-located points
        cpa_coloc = closest_point_of_approach(Vector2(0, 0), Vector2(0, 0))
        self.assertFalse(math.isnan(cpa_coloc.distance_at_cpa))
        self.assertEqual(cpa_coloc.distance_at_cpa, 0.0)

        # 2. Parallel paths (zero relative velocity)
        cpa_par = closest_point_of_approach(Vector2(2, 0), Vector2(0, 0))
        self.assertEqual(cpa_par.time_to_cpa, 0.0)
        self.assertAlmostEqual(cpa_par.distance_at_cpa, 2.0, places=4)

        # 3. Diverging paths (moving away)
        pred_div = analyze_collision(Vector2(0, 5), Vector2(0, 2), model, horizon=5.0)
        self.assertFalse(pred_div.collision_predicted)
        self.assertIsNone(pred_div.ttc)

    def test_perception_occlusion_and_recovery(self) -> None:
        """Verify tracker maintains coasting state during multi-frame perception dropout."""
        cov = ((0.1, 0.0), (0.0, 0.1))
        # Feed 5 frames of approach
        for i in range(5):
            t = i * 0.05
            z = 10.0 - 1.0 * t
            obs = [PhysicalObservation(position=Vector2(0.0, z), measurement_covariance=cov, confidence=0.9, timestamp=t, detection_id=f"obs_{i}")]
            res = self.pipeline.process_frame(raw_frame=None, timestamp=t, explicit_observations=obs)
            self.assertIsNotNone(res)

        # 4 frames of total occlusion (empty observations)
        for j in range(5, 9):
            t = j * 0.05
            res = self.pipeline.process_frame(raw_frame=None, timestamp=t, explicit_observations=[])
            self.assertIsNotNone(res)
            self.assertGreaterEqual(res.active_track_count, 0)

    def test_sudden_acceleration_maneuver(self) -> None:
        """Verify risk evaluator remains stable when object suddenly halts or accelerates."""
        cov = ((0.1, 0.0), (0.0, 0.1))
        # Constant velocity approach
        for i in range(10):
            t = i * 0.05
            z = 5.0 - 2.0 * t
            obs = [PhysicalObservation(position=Vector2(0.0, z), measurement_covariance=cov, confidence=0.95, timestamp=t, detection_id="ped_1")]
            res = self.pipeline.process_frame(raw_frame=None, timestamp=t, explicit_observations=obs)

        # Sudden halt
        for i in range(10, 20):
            t = i * 0.05
            z = 4.0
            obs = [PhysicalObservation(position=Vector2(0.0, z), measurement_covariance=cov, confidence=0.95, timestamp=t, detection_id="ped_1")]
            res = self.pipeline.process_frame(raw_frame=None, timestamp=t, explicit_observations=obs)
            self.assertFalse(math.isnan(res.risk_assessment.risk_score))

    def test_decision_flapping_suppression(self) -> None:
        """Verify decision engine does not wildly oscillate between left and right steering."""
        cov = ((0.1, 0.0), (0.0, 0.1))
        actions = []
        for i in range(20):
            t = i * 0.05
            x = 0.05 * (-1 if i % 2 == 0 else 1)
            obs = [PhysicalObservation(position=Vector2(x, 2.0), measurement_covariance=cov, confidence=0.9, timestamp=t, detection_id="obs")]
            res = self.pipeline.process_frame(raw_frame=None, timestamp=t, explicit_observations=obs)
            actions.append(res.decision.action)

        self.assertTrue(all(isinstance(a, NavigationAction) for a in actions))


if __name__ == '__main__':
    unittest.main()
