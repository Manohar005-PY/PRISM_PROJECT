"""Unit tests for RealtimePRISMPipeline lifecycle and execution."""

import unittest

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
from app.runtime.realtime.camera import CameraConfig, CameraFrame, MockCameraSource
from app.runtime.realtime.config import RealtimeConfig
from app.runtime.realtime.inference import MockInferenceEngine
from app.runtime.realtime.pipeline import RealtimePRISMPipeline
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator
from app.state.projection import CameraIntrinsics, CameraPose, GroundPlaneProjector, ProjectionConfig
from app.tracking.config import TrackingConfig
from app.tracking.tracker import MultiObjectTracker


class TestRealtimePipeline(unittest.TestCase):
    """Test suite for RealtimePRISMPipeline."""

    def setUp(self) -> None:
        intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
        pose = CameraPose(1.2, 0.15)
        proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
        self.projector = GroundPlaneProjector(proj_cfg)

        state_estimator = KalmanStateEstimator(KalmanFilterConfig())
        tracker = MultiObjectTracker(TrackingConfig(), state_estimator=state_estimator)
        detector = SyntheticDetector(config=proj_cfg, pixel_noise_sigma=1.0, random_seed=42)
        perc_pipe = PerceptionPipeline(detector, self.projector)

        self.core_pipeline = PRISMPipeline(
            perception_pipeline=perc_pipe,
            tracker=tracker,
            state_estimator=state_estimator,
            predictor=ConstantVelocityPredictor(),
            risk_evaluator=PredictiveRiskEvaluator(RiskConfig()),
            decision_engine=DecisionEngine(),
            config=RuntimeConfig(),
        )

    def test_realtime_pipeline_lifecycle(self) -> None:
        cam = MockCameraSource(CameraConfig())
        engine = MockInferenceEngine([])

        pipeline = RealtimePRISMPipeline(
            camera_source=cam,
            inference_engine=engine,
            projector=self.projector,
            core_pipeline=self.core_pipeline,
            config=RealtimeConfig(),
        )

        self.assertFalse(pipeline.is_running)
        self.assertTrue(pipeline.start())
        self.assertTrue(pipeline.is_running)

        res1 = pipeline.step()
        self.assertIsNotNone(res1)
        res2 = pipeline.step()
        self.assertIsNotNone(res2)

        metrics = pipeline.get_metrics()
        self.assertGreater(metrics.mean_latency_ms, 0.0)

        pipeline.stop()
        self.assertFalse(pipeline.is_running)


if __name__ == "__main__":
    unittest.main()
