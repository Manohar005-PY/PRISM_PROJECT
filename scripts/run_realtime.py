"""PRISM Live Real-Time Camera Runtime Entry Point.

Executes continuous live camera capture, YOLOv8 inference, PRISM temporal state estimation,
prediction, risk evaluation, and spatial avoidance decision planning.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.decision.engine import DecisionEngine
from app.perception.adapters.opencv_camera import OpenCVCameraAdapter
from app.runtime.frame_loop import FrameLoop
from app.runtime.pipeline import PRISMPipeline
from app.runtime.realtime.camera import MockCameraSource
from app.runtime.realtime.inference import MockInferenceEngine, YOLOInferenceEngine
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
from app.prediction.constant_velocity import ConstantVelocityPredictor
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import RiskConfig
from app.runtime.config import RuntimeConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="PRISM Real-Time Camera Runtime")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO weights")
    parser.add_argument("--mock", action="store_true", help="Force mock camera and detector")
    parser.add_argument("--max-frames", type=int, default=30, help="Max frames to process (0 = infinite)")
    parser.add_argument("--log-file", type=str, default="telemetry_log.jsonl", help="JSONL telemetry log path")
    args = parser.parse_args()

    print("=" * 80)
    print(" PRISM REAL-TIME CAMERA RUNTIME")
    print("=" * 80)

    from app.runtime.realtime.config import CameraConfig
    cam_cfg = CameraConfig(device_index=args.camera_id, width=args.width, height=args.height, target_fps=args.fps)

    # 1. Setup Camera
    if args.mock:
        print("Initializing Mock Camera Source...")
        cam_source = MockCameraSource(config=cam_cfg)
    else:
        print(f"Initializing OpenCV Camera Source on device {args.camera_id}...")
        cam_source = OpenCVCameraAdapter(device_index=args.camera_id, width=args.width, height=args.height, target_fps=args.fps)
        if not cam_source.open():
            print("WARNING: OpenCV/Camera initialization failed. Falling back to Mock Camera Source.")
            cam_source = MockCameraSource(config=cam_cfg)

    # 2. Setup Inference Engine
    if args.mock:
        inf_engine = MockInferenceEngine()
    else:
        try:
            inf_engine = YOLOInferenceEngine()
            if not inf_engine.is_available():
                print("WARNING: YOLO/Torch not available. Falling back to Mock Inference Engine.")
                inf_engine = MockInferenceEngine()
        except Exception:
            inf_engine = MockInferenceEngine()

    # 3. Setup Projector & Core Pipeline
    intrinsics = CameraIntrinsics(args.width, args.height, 500.0, 500.0, args.width / 2.0, args.height / 2.0)
    pose = CameraPose(height=1.2, pitch=0.15)
    proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
    projector = GroundPlaneProjector(proj_cfg)

    state_estimator = KalmanStateEstimator(KalmanFilterConfig())
    tracker = MultiObjectTracker(TrackingConfig(), state_estimator=state_estimator)
    from app.perception.adapters.synthetic import SyntheticDetector
    from app.perception.pipeline import PerceptionPipeline
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

    telemetry = TelemetryLogger(log_filepath=args.log_file)
    visualizer = DebugVisualizer(enabled=True)

    frame_loop = FrameLoop(
        camera_source=cam_source,
        inference_engine=inf_engine,
        projector=projector,
        core_pipeline=core_pipeline,
        telemetry_logger=telemetry,
        visualizer=visualizer,
        target_fps=args.fps,
    )

    print("\nStarting Real-Time Frame Loop...")
    print(f"{'FRAME':<6} | {'TIME':<6} | {'ACTIVE TRACKS':<14} | {'RISK LEVEL':<10} | {'ACTION':<10} | {'LATENCY'}")
    print("-" * 80)

    def frame_callback(result, annotated_frame) -> bool:
        t_str = f"{result.timestamp:.2f}s"
        risk_str = result.risk_assessment.risk_level.value if result.risk_assessment else "SAFE"
        act_str = result.decision.action.value
        lat_str = f"{result.processing_duration_ms:.2f}ms"
        print(f"{result.frame_id:<6} | {t_str:<6} | {result.active_track_count:<14} | {risk_str:<10} | {act_str:<10} | {lat_str}")
        return True

    max_f = args.max_frames if args.max_frames > 0 else None
    frame_loop.run_loop(max_frames=max_f, callback=frame_callback)

    aggr = telemetry.compute_aggregates()
    print("-" * 80)
    print("SESSION SUMMARY:")
    print(f" - Total Frames Processed: {aggr.total_frames}")
    print(f" - Mean FPS:               {aggr.mean_fps:.1f}")
    print(f" - Mean Latency:           {aggr.mean_latency_ms:.2f} ms")
    print(f" - P95 Latency:            {aggr.p95_latency_ms:.2f} ms")
    print(f" - Decision Switches:      {aggr.decision_switch_count}")
    print(f" - Telemetry Log Saved:    {args.log_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
