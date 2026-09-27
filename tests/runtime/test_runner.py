"""Unit tests for PRISMRunner frame loop execution."""

import unittest

from app.core.types import Vector2
from app.perception.adapters.synthetic import SyntheticDetector
from app.perception.pipeline import PerceptionPipeline
from app.prediction.constant_velocity import ConstantVelocityPredictor
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import RiskConfig
from app.runtime.config import RuntimeConfig
from app.runtime.pipeline import PRISMPipeline
from app.runtime.runner import PRISMRunner
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator
from app.state.projection import CameraIntrinsics, CameraPose, GroundPlaneProjector, ProjectionConfig
from app.tracking.config import TrackingConfig
from app.tracking.tracker import MultiObjectTracker


class TestPRISMRunner(unittest.TestCase):
    """Test suite for PRISMRunner sequence processor."""

    def test_run_sequence(self) -> None:
        intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
        pose = CameraPose(1.2, 0.15)
        proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
        projector = GroundPlaneProjector(proj_cfg)
        detector = SyntheticDetector(config=proj_cfg, pixel_noise_sigma=1.0, random_seed=42)
        perc_pipe = PerceptionPipeline(detector, projector)

        state_estimator = KalmanStateEstimator(KalmanFilterConfig())
        tracker = MultiObjectTracker(TrackingConfig(), state_estimator=state_estimator)

        pipeline = PRISMPipeline(
            perception_pipeline=perc_pipe,
            tracker=tracker,
            state_estimator=state_estimator,
            config=RuntimeConfig(),
        )

        runner = PRISMRunner(pipeline)

        # 5 synthetic frames
        frames = [([(1, "person", Vector2(0.0, 4.0 - i * 0.2))], i * 0.1) for i in range(5)]
        results = runner.run_sequence(frames)

        self.assertEqual(len(results), 5)
        self.assertEqual(results[-1].frame_id, 5)


if __name__ == "__main__":
    unittest.main()
