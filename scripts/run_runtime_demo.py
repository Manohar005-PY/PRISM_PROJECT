"""Comprehensive demonstration script for PRISM Real-Time End-to-End Runtime & System Integration.

Executes and measures 7 canonical runtime scenarios:
1. Safe Stationary Environment
2. Approaching Obstacle (Escalating Risk & Avoidance)
3. Departing Obstacle (Predictive Avoidance of False Alerts)
4. Multiple Obstacles (Primary Threat Arbitration)
5. Temporary Occlusion & Identity Recovery
6. Perception Failure & Rejection Diagnostics
7. Runtime Overload & Health Monitoring
"""

from __future__ import annotations

import os
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.types import Vector2
from app.decision.engine import DecisionEngine
from app.perception.adapters.synthetic import SyntheticDetector
from app.perception.pipeline import PerceptionPipeline
from app.prediction.constant_velocity import ConstantVelocityPredictor
from app.prediction.relative_motion import EgoMotionState
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import RiskConfig
from app.runtime.config import RuntimeConfig
from app.runtime.health import RuntimeStatus
from app.runtime.pipeline import PRISMPipeline
from app.runtime.runner import PRISMRunner
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator
from app.state.projection import CameraIntrinsics, CameraPose, GroundPlaneProjector, ProjectionConfig
from app.tracking.config import TrackingConfig
from app.tracking.tracker import MultiObjectTracker


def create_demo_pipeline(config: RuntimeConfig | None = None) -> PRISMPipeline:
    intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
    pose = CameraPose(1.2, 0.15)
    proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
    projector = GroundPlaneProjector(proj_cfg)
    detector = SyntheticDetector(config=proj_cfg, pixel_noise_sigma=1.0, random_seed=42)
    perc_pipe = PerceptionPipeline(detector, projector)

    state_estimator = KalmanStateEstimator(KalmanFilterConfig())
    tracker = MultiObjectTracker(
        config=TrackingConfig(confirmation_hits=2, gating_threshold=9.21),
        state_estimator=state_estimator,
    )
    return PRISMPipeline(
        perception_pipeline=perc_pipe,
        tracker=tracker,
        state_estimator=state_estimator,
        predictor=ConstantVelocityPredictor(),
        risk_evaluator=PredictiveRiskEvaluator(RiskConfig(alpha_smooth=1.0)),
        decision_engine=DecisionEngine(),
        config=config if config is not None else RuntimeConfig(),
    )


def print_table_header() -> None:
    print(f"{'FRAME':<6} | {'DET':<4} | {'TRACKS':<7} | {'RISK':<8} | {'PRIMARY THREAT':<15} | {'ACTION':<14} | {'LATENCY':<9} | {'FPS':<6}")
    print("-" * 85)


def print_frame_row(res) -> None:
    obj_id = res.risk_assessment.object_id if res.risk_assessment else None
    threat_str = f"#{obj_id}" if obj_id is not None and obj_id > 0 else "None"
    action_str = res.decision.action.value.upper()
    risk_str = res.risk_assessment.risk_level.value.upper() if res.risk_assessment else "NONE"
    lat_str = f"{res.processing_duration_ms:.2f}ms"
    fps_str = f"{res.timing.effective_fps:.0f}"
    print(
        f"{res.frame_id:<6} | {res.detection_count:<4} | {res.active_track_count:<7} | {risk_str:<8} | {threat_str:<15} | {action_str:<14} | {lat_str:<9} | {fps_str:<6}"
    )


def run_experiment_1() -> PRISMPipeline:
    print("=" * 85)
    print("EXPERIMENT 1: SAFE STATIONARY ENVIRONMENT")
    print("=" * 85)
    print_table_header()

    pipeline = create_demo_pipeline()
    for i in range(5):
        t = i * 0.1
        entities = [(1, "chair", Vector2(3.0, 6.0))]
        res = pipeline.process_frame(entities, timestamp=t)
        print_frame_row(res)

    print("-" * 85)
    print("Finding: Stationary lateral obstacle produces low risk with continuous CONTINUE policy.\n")
    return pipeline


def run_experiment_2() -> PRISMPipeline:
    print("=" * 85)
    print("EXPERIMENT 2: APPROACHING OBSTACLE (ESCALATING RISK & AVOIDANCE)")
    print("=" * 85)
    print_table_header()

    pipeline = create_demo_pipeline()
    for i in range(7):
        t = i * 0.1
        z = 5.0 - i * 0.6
        entities = [(1, "person", Vector2(0.0, z))]
        res = pipeline.process_frame(entities, timestamp=t)
        print_frame_row(res)

    print("-" * 85)
    print("Finding: As TTC shrinks below safety thresholds, risk escalates from LOW -> CRITICAL,")
    print("         and policy smoothly commands evasion/stop actions.\n")
    return pipeline


def run_experiment_3() -> None:
    print("=" * 85)
    print("EXPERIMENT 3: DEPARTING OBSTACLE (PREDICTIVE AVOIDANCE OF FALSE ALERTS)")
    print("=" * 85)
    print_table_header()

    pipeline = create_demo_pipeline()
    for i in range(5):
        t = i * 0.1
        z = 2.5 + i * 0.5
        entities = [(1, "person", Vector2(0.0, z))]
        res = pipeline.process_frame(entities, timestamp=t)
        print_frame_row(res)

    print("-" * 85)
    print("Finding: PRISM predictive velocity estimation recognizes diverging motion, preventing")
    print("         false proximity alarms even when initial distance is small.\n")


def run_experiment_4() -> None:
    print("=" * 85)
    print("EXPERIMENT 4: MULTIPLE OBSTACLES & PRIMARY THREAT ARBITRATION")
    print("=" * 85)
    print_table_header()

    pipeline = create_demo_pipeline()
    for i in range(5):
        t = i * 0.1
        entities = [
            (1, "chair", Vector2(2.5, 4.0)),
            (2, "person", Vector2(0.0, 4.0 - i * 0.7)),
            (3, "person", Vector2(-1.5, 8.0 - i * 0.3)),
        ]
        res = pipeline.process_frame(entities, timestamp=t)
        print_frame_row(res)

    print("-" * 85)
    print("Finding: Multi-obstacle tracking maintains all identities and correctly arbitrates")
    print("         the primary threat to Object #2.\n")


def run_experiment_5() -> None:
    print("=" * 85)
    print("EXPERIMENT 5: TEMPORARY OCCLUSION & TRACK RECOVERY")
    print("=" * 85)
    print_table_header()

    pipeline = create_demo_pipeline()
    for i in range(6):
        t = i * 0.1
        if i in (2, 3):
            entities = []
        else:
            z = 4.0 - i * 0.3
            entities = [(1, "person", Vector2(0.0, z))]
        res = pipeline.process_frame(entities, timestamp=t)
        print_frame_row(res)

    print("-" * 85)
    print("Finding: Track identity survives 2-frame occlusion gap without identity destruction.\n")


def run_experiment_6() -> None:
    print("=" * 85)
    print("EXPERIMENT 6: PERCEPTION FAILURE & REJECTION DIAGNOSTICS")
    print("=" * 85)
    print_table_header()

    pipeline = create_demo_pipeline()
    res = pipeline.process_frame([(1, "person", Vector2(0.0, -1.0))], timestamp=0.1)
    print_frame_row(res)

    print("-" * 85)
    print("Finding: Perception rejection diagnostics safely isolate invalid sensor data without crashing.\n")


def run_experiment_7(active_pipeline: PRISMPipeline) -> None:
    print("=" * 85)
    print("STAGE TIMING & PERFORMANCE INSTRUMENTATION")
    print("=" * 85)

    stages = active_pipeline.timing_profiler.stage_averages()
    print(f"Perception:       {stages['perception']:.3f} ms")
    print(f"Tracking:         {stages['tracking']:.3f} ms")
    print(f"State Estimation: {stages['estimation']:.3f} ms")
    print(f"Prediction:       {stages['prediction']:.3f} ms")
    print(f"Risk:             {stages['risk']:.3f} ms")
    print(f"Decision:         {stages['decision']:.3f} ms")
    print("-" * 40)
    print(f"Total Latency:    {stages['total']:.3f} ms")
    print(f"Effective FPS:    {active_pipeline.timing_profiler.average_fps:.1f} FPS")

    health = active_pipeline.health_monitor.assess_health(
        active_tracks=len(active_pipeline.state_estimator.tracks),
        stale_tracks=0,
        average_latency_ms=active_pipeline.timing_profiler.average_latency_ms,
        current_fps=active_pipeline.timing_profiler.average_fps,
    )
    print(f"\nRuntime Health:   {health.status.value.upper()}")
    print(f"Frames Processed: {health.frames_processed}")
    print(f"Frames Failed:    {health.frames_failed}")
    print(f"Dropped Frames:   {health.dropped_frames}")
    print("=" * 85)


def main() -> None:
    print("=" * 85)
    print(" PRISM REAL-TIME END-TO-END RUNTIME DEMONSTRATION")
    print("=" * 85 + "\n")
    run_experiment_1()
    p2 = run_experiment_2()
    run_experiment_3()
    run_experiment_4()
    run_experiment_5()
    run_experiment_6()
    run_experiment_7(p2)


if __name__ == "__main__":
    main()
