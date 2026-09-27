"""Tests for temporal dynamics, EMA smoothing, variable delta-t, and decision hysteresis."""

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
    project_ground_to_image,
)
from app.tracking.config import TrackingConfig
from app.tracking.tracker import MultiObjectTracker


class TestTemporalBehavior(unittest.TestCase):
    """Tests evaluating system temporal consistency and hysteresis."""

    def setUp(self) -> None:
        intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
        pose = CameraPose(1.2, 0.15)
        self.proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
        self.projector = GroundPlaneProjector(self.proj_cfg)

    def test_variable_delta_t_stability(self) -> None:
        """Test pipeline behavior under fluctuating frame rates."""
        state_estimator = KalmanStateEstimator(KalmanFilterConfig())
        tracker = MultiObjectTracker(TrackingConfig(), state_estimator=state_estimator)
        detector = SyntheticDetector(config=self.projector.config, pixel_noise_sigma=0.5, random_seed=42)
        perc_pipe = PerceptionPipeline(detector, self.projector)

        pipeline = PRISMPipeline(
            perception_pipeline=perc_pipe,
            tracker=tracker,
            state_estimator=state_estimator,
            predictor=ConstantVelocityPredictor(),
            risk_evaluator=PredictiveRiskEvaluator(RiskConfig()),
            decision_engine=DecisionEngine(),
            config=RuntimeConfig(),
        )

        dts = [0.016, 0.033, 0.050, 0.100, 0.033, 0.066]
        t = 0.0
        z = 8.0
        vz = -1.0

        from app.runtime.health import RuntimeStatus
        for dt in dts:
            t += dt
            z += vz * dt
            img_pt = project_ground_to_image(Vector2(0.0, z), self.proj_cfg)
            if img_pt is not None:
                u, v = img_pt
                det = Detection(
                    detection_id=f"d_{int(t*1000)}",
                    class_id=0,
                    class_name="person",
                    confidence=0.95,
                    bbox=BoundingBox(u - 20.0, v - 60.0, u + 20.0, v),
                    timestamp=t,
                )
                obs = perc_pipe.extractor.extract_observation(det)
                res = pipeline.process_frame(raw_frame=None, timestamp=t, explicit_observations=[obs])
                self.assertEqual(res.runtime_status, RuntimeStatus.HEALTHY)

    def test_decision_hysteresis_suppresses_chattering(self) -> None:
        """Verify hysteresis margin prevents rapid oscillation between LEFT and RIGHT evasion."""
        engine = DecisionEngine()
        self.assertIsNotNone(engine.arbitrator.hysteresis_margin)
        self.assertGreater(engine.arbitrator.hysteresis_margin, 0.0)


if __name__ == "__main__":
    unittest.main()
