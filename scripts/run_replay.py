"""PRISM Deterministic Dataset Replay CLI.

Replays recorded detection JSONL datasets through the PRISM pipeline without re-running ML inference.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.types import Vector2
from app.decision.engine import DecisionEngine
from app.perception.adapters.synthetic import SyntheticDetector
from app.perception.observation import ObservationExtractor
from app.perception.pipeline import PerceptionPipeline
from app.prediction.constant_velocity import ConstantVelocityPredictor
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import RiskConfig
from app.runtime.config import RuntimeConfig
from app.runtime.pipeline import PRISMPipeline
from app.runtime.recorder import FrameRecorder
from app.runtime.replay import ReplayFrameSource
from app.runtime.telemetry import TelemetryLogger
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
from app.perception.model import Detection, PhysicalObservation
from app.core.types import BoundingBox


def create_sample_recording(filepath: str) -> None:
    """Helper to generate a sample recording file if none exists."""
    recorder = FrameRecorder(filepath)
    intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
    pose = CameraPose(1.2, 0.15)
    proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)

    for i in range(15):
        t = i * 0.1
        z = 5.0 - 0.8 * t
        img_pt = project_ground_to_image(Vector2(0.0, z), proj_cfg)
        if img_pt is not None:
            u, v = img_pt
            det = Detection(
                detection_id=f"rec_det_{i}",
                class_id=0,
                class_name="person",
                confidence=0.92,
                bbox=BoundingBox(u - 20.0, v - 60.0, u + 20.0, v),
                timestamp=t,
            )
            recorder.record_frame_detections(frame_id=i + 1, timestamp=t, detections=[det])


def main() -> None:
    parser = argparse.ArgumentParser(description="PRISM Dataset Replay Tool")
    parser.add_argument("--input", type=str, default="sample_recording.jsonl", help="Input JSONL recording path")
    parser.add_argument("--create-sample", action="store_true", help="Generate a sample recording before replay")
    args = parser.parse_args()

    print("=" * 80)
    print(" PRISM DETERMINISTIC DATASET REPLAY")
    print("=" * 80)

    if args.create_sample or not os.path.exists(args.input):
        print(f"Creating sample recording at: {args.input}")
        create_sample_recording(args.input)

    replay_source = ReplayFrameSource(args.input)
    print(f"Loaded {replay_source.total_frames} recorded frames from: {args.input}")

    # Setup PRISM Pipeline
    intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
    pose = CameraPose(1.2, 0.15)
    proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
    projector = GroundPlaneProjector(proj_cfg)
    extractor = ObservationExtractor(projector=projector)

    state_estimator = KalmanStateEstimator(KalmanFilterConfig())
    tracker = MultiObjectTracker(TrackingConfig(), state_estimator=state_estimator)
    perc_pipe = PerceptionPipeline(SyntheticDetector(config=proj_cfg), projector)

    core_pipeline = PRISMPipeline(
        perception_pipeline=perc_pipe,
        tracker=tracker,
        state_estimator=state_estimator,
        predictor=ConstantVelocityPredictor(),
        risk_evaluator=PredictiveRiskEvaluator(RiskConfig()),
        decision_engine=DecisionEngine(),
        config=RuntimeConfig(),
    )

    telemetry = TelemetryLogger()

    print("\nReplaying Frames through PRISM Pipeline...")
    print(f"{'FRAME':<6} | {'TIME':<6} | {'OBS COUNT':<10} | {'ACTIVE TRACKS':<14} | {'RISK LEVEL':<10} | {'ACTION'}")
    print("-" * 80)

    for frame_id, timestamp, detections in replay_source:
        obs_list = []
        for d in detections:
            res_obs = extractor.extract_observation(d)
            if isinstance(res_obs, PhysicalObservation):
                obs_list.append(res_obs)

        result = core_pipeline.process_frame(raw_frame=None, timestamp=timestamp, explicit_observations=obs_list)
        telemetry.record_frame(result)

        t_str = f"{timestamp:.2f}s"
        risk_str = result.risk_assessment.risk_level.value if result.risk_assessment else "SAFE"
        act_str = result.decision.action.value
        print(f"{frame_id:<6} | {t_str:<6} | {len(obs_list):<10} | {result.active_track_count:<14} | {risk_str:<10} | {act_str}")

    aggr = telemetry.compute_aggregates()
    print("-" * 80)
    print("REPLAY COMPLETE:")
    print(f" - Frames Processed: {aggr.total_frames}")
    print(f" - Mean Latency:    {aggr.mean_latency_ms:.2f} ms")
    print(f" - Decision Switches: {aggr.decision_switch_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
