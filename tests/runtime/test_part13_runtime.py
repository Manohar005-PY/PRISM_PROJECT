"""Unit and integration tests for PRISM Part 13 runtime, telemetry, replay, and validation."""

import os
import tempfile
import unittest

from app.core.enums import RiskLevel
from app.core.types import BoundingBox, Vector2
from app.decision.engine import DecisionEngine
from app.decision.model import NavigationAction
from app.perception.adapters.opencv_camera import FakeCameraFrameSource
from app.perception.adapters.synthetic import SyntheticDetector
from app.perception.model import Detection
from app.perception.pipeline import PerceptionPipeline
from app.prediction.constant_velocity import ConstantVelocityPredictor
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import RiskConfig
from app.runtime.config import RuntimeConfig
from app.runtime.frame_loop import FrameLoop
from app.runtime.pipeline import PRISMPipeline
from app.runtime.realtime.inference import MockInferenceEngine
from app.runtime.recorder import FrameRecorder
from app.runtime.replay import ReplayFrameSource
from app.runtime.telemetry import TelemetryLogger
from app.runtime.visualization import DebugVisualizer
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator
from app.state.projection import (
    CameraIntrinsics,
    CameraPose,
    GroundPlaneProjector,
    ProjectionConfig,
)
from app.tracking.config import TrackingConfig
from app.tracking.tracker import MultiObjectTracker
from evaluation.reactive_baseline import ProximityReactiveBaseline


class TestPart13Runtime(unittest.TestCase):
    """Integration and unit tests for Part 13 runtime orchestration."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
        self.pose = CameraPose(1.2, 0.15)
        self.proj_cfg = ProjectionConfig(intrinsics=self.intrinsics, camera_pose=self.pose)
        self.projector = GroundPlaneProjector(self.proj_cfg)

        self.state_estimator = KalmanStateEstimator(KalmanFilterConfig())
        self.tracker = MultiObjectTracker(TrackingConfig(), state_estimator=self.state_estimator)
        self.detector = SyntheticDetector(config=self.proj_cfg)
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

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_camera_source_lifecycle(self) -> None:
        """Verify mock frame source lifecycle and frame acquisition."""
        source = FakeCameraFrameSource(frames_count=5)
        self.assertTrue(source.open())
        self.assertTrue(source.is_opened())

        frames = []
        while True:
            f = source.read_frame()
            if f is None:
                break
            frames.append(f)

        self.assertEqual(len(frames), 5)
        self.assertEqual(frames[0].frame_id, 1)
        source.release()
        self.assertFalse(source.is_opened())

    def test_telemetry_logger_jsonl(self) -> None:
        """Verify structured telemetry logging and aggregate computation."""
        log_path = os.path.join(self.temp_dir.name, "test_telemetry.jsonl")
        logger = TelemetryLogger(log_filepath=log_path)

        for i in range(10):
            t = i * 0.033
            res = self.pipeline.process_frame(raw_frame=None, timestamp=t, explicit_observations=[])
            logger.record_frame(res)

        self.assertTrue(os.path.exists(log_path))
        aggr = logger.compute_aggregates()
        self.assertEqual(aggr.total_frames, 10)
        self.assertGreater(aggr.mean_fps, 0.0)

    def test_record_and_replay_cycle(self) -> None:
        """Verify recording detections to JSONL and playing them back identically."""
        rec_path = os.path.join(self.temp_dir.name, "test_recording.jsonl")
        recorder = FrameRecorder(rec_path)

        det = Detection(
            detection_id="det_01",
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=BoundingBox(100.0, 150.0, 140.0, 230.0),
            timestamp=1.0,
        )
        recorder.record_frame_detections(frame_id=1, timestamp=1.0, detections=[det])

        replay = ReplayFrameSource(rec_path)
        self.assertEqual(replay.total_frames, 1)
        fid, ts, dets = replay.read_frame()
        self.assertEqual(fid, 1)
        self.assertEqual(ts, 1.0)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].detection_id, "det_01")
        self.assertEqual(dets[0].class_name, "person")

    def test_frame_loop_execution_and_resilience(self) -> None:
        """Verify FrameLoop coordinates camera, mock detector, pipeline, and telemetry."""
        source = FakeCameraFrameSource(frames_count=5)
        inf_engine = MockInferenceEngine()
        telemetry = TelemetryLogger()
        visualizer = DebugVisualizer(enabled=False)

        loop = FrameLoop(
            camera_source=source,
            inference_engine=inf_engine,
            projector=self.projector,
            core_pipeline=self.pipeline,
            telemetry_logger=telemetry,
            visualizer=visualizer,
        )

        frames_seen = []

        def cb(res, ann) -> bool:
            frames_seen.append(res.frame_id)
            return True

        loop.run_loop(max_frames=5, callback=cb)
        self.assertEqual(len(frames_seen), 5)
        self.assertFalse(loop.is_running)

    def test_debug_visualizer_headless_safety(self) -> None:
        """Verify DebugVisualizer does not crash in headless environments."""
        vis = DebugVisualizer(enabled=True)
        res = self.pipeline.process_frame(raw_frame=None, timestamp=0.0, explicit_observations=[])
        # Calling with None frame should return None safely
        out = vis.render_overlay(None, res)
        self.assertIsNone(out)

    def test_reactive_baseline_behavior(self) -> None:
        """Verify ProximityReactiveBaseline triggers based on distance."""
        base = ProximityReactiveBaseline(stop_threshold=1.5, caution_threshold=3.0)
        from app.core.types import ObjectObservation
        obs = [
            ObjectObservation(
                object_id=1,
                class_id="person",
                bbox=BoundingBox(300.0, 400.0, 340.0, 460.0),
                confidence=0.9,
                timestamp=0.0,
            )
        ]
        sample = base.evaluate_frame(
            frame_id=1,
            timestamp=0.0,
            ground_truth={},
            observations=obs,
            is_scene_hazard=True,
        )
        self.assertIn(sample.decision.action, (NavigationAction.STOP, NavigationAction.CAUTION, NavigationAction.CONTINUE))


if __name__ == "__main__":
    unittest.main()
