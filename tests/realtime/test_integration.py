"""End-to-end integration test for the Realtime subsystem executing closed-loop multi-frame scenarios."""

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


class TestRealtimeIntegration(unittest.TestCase):
    """Integration test suite executing multi-frame sequence through RealtimePRISMPipeline."""

    def test_full_realtime_pipeline_integration(self) -> None:
        intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
        pose = CameraPose(1.2, 0.15)
        proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
        projector = GroundPlaneProjector(proj_cfg)

        state_estimator = KalmanStateEstimator(KalmanFilterConfig())
        tracker = MultiObjectTracker(TrackingConfig(confirmation_hits=2), state_estimator=state_estimator)
        detector = SyntheticDetector(config=proj_cfg, pixel_noise_sigma=1.0, random_seed=42)
        perc_pipe = PerceptionPipeline(detector, projector)

        core_pipeline = PRISMPipeline(
            perception_pipeline=perc_pipe,
            tracker=tracker,
            state_estimator=state_estimator,
            predictor=ConstantVelocityPredictor(),
            risk_evaluator=PredictiveRiskEvaluator(RiskConfig()),
            decision_engine=DecisionEngine(),
            config=RuntimeConfig(),
        )

        # Sequence of 5 frames of an approaching pedestrian
        dets_seq = []
        for i in range(5):
            # bottom center y increases (approaching camera)
            # v = 250 + i * 20
            d = Detection(
                detection_id=f"det_{i}",
                class_id=0,
                class_name="person",
                confidence=0.92,
                bbox=BoundingBox(300.0, 200.0 + i * 15, 340.0, 250.0 + i * 20),
                timestamp=i * 0.1,
            )
            dets_seq.append([d])

        cam = MockCameraSource(CameraConfig())
        engine = MockInferenceEngine(dets_seq)

        pipeline = RealtimePRISMPipeline(
            camera_source=cam,
            inference_engine=engine,
            projector=projector,
            core_pipeline=core_pipeline,
            config=RealtimeConfig(),
        )

        self.assertTrue(pipeline.start())

        results = []
        for i in range(5):
            res = pipeline.step()
            if res is not None:
                results.append(res)

        pipeline.stop()

        self.assertEqual(len(results), 5)
        metrics = pipeline.get_metrics()
        self.assertGreater(metrics.processed_fps, 0.0)
        self.assertEqual(metrics.dropped_frames, 0)


if __name__ == "__main__":
    unittest.main()
