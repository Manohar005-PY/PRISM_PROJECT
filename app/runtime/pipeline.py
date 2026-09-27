"""PRISMPipeline: Synchronous orchestrator executing Perception -> Tracking -> Estimation -> Prediction -> Risk -> Decision."""

from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence

from app.core.enums import RiskLevel
from app.core.types import ObjectState, RiskAssessment, Trajectory, Vector2, WorldState
from app.decision.context import DecisionContext
from app.decision.engine import DecisionEngine
from app.decision.model import DecisionReason, NavigationAction, NavigationDecision
from app.perception.model import Detection, DetectionRejection, PhysicalObservation, TrackedObservation
from app.perception.pipeline import PerceptionPipeline
from app.prediction.constant_velocity import ConstantVelocityPredictor
from app.prediction.relative_motion import EgoMotionState
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import RiskConfig
from app.runtime.config import RuntimeConfig
from app.runtime.context import ContextBuilder
from app.runtime.errors import (
    create_fail_safe_decision,
    create_fail_safe_risk_assessment,
)
from app.runtime.health import HealthMonitor, RuntimeStatus
from app.runtime.model import FrameContext, RuntimeResult
from app.runtime.timing import RollingTimingProfiler, StageTiming
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator
from app.tracking.config import TrackingConfig
from app.tracking.tracker import MultiObjectTracker


class PRISMPipeline:
    """End-to-end real-time runtime pipeline executing PRISM stages 1 through 9.

    Orchestrates the lifecycle of each frame with failure isolation, timing profiling,
    stale-data detection, and fail-safe policy enforcement.
    """

    def __init__(
        self,
        perception_pipeline: PerceptionPipeline,
        tracker: MultiObjectTracker | None = None,
        state_estimator: KalmanStateEstimator | None = None,
        predictor: ConstantVelocityPredictor | None = None,
        risk_evaluator: PredictiveRiskEvaluator | None = None,
        decision_engine: DecisionEngine | None = None,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.config: RuntimeConfig = config if config is not None else RuntimeConfig()
        self.perception_pipeline: PerceptionPipeline = perception_pipeline

        self.state_estimator: KalmanStateEstimator = (
            state_estimator if state_estimator is not None else KalmanStateEstimator(config=KalmanFilterConfig())
        )
        self.tracker: MultiObjectTracker = (
            tracker
            if tracker is not None
            else MultiObjectTracker(config=TrackingConfig(), state_estimator=self.state_estimator)
        )
        self.predictor: ConstantVelocityPredictor = (
            predictor if predictor is not None else ConstantVelocityPredictor()
        )
        self.risk_evaluator: PredictiveRiskEvaluator = (
            risk_evaluator if risk_evaluator is not None else PredictiveRiskEvaluator(RiskConfig())
        )
        self.decision_engine: DecisionEngine = (
            decision_engine if decision_engine is not None else DecisionEngine()
        )

        self.health_monitor = HealthMonitor(self.config)
        self.timing_profiler = RollingTimingProfiler(window_size=50)

        self._frame_counter: int = 0
        self._last_frame_timestamp: float | None = None
        self._last_observation_timestamp: float | None = None

    def reset(self) -> None:
        """Reset runtime pipeline state, trackers, filters, and health monitors."""
        self._frame_counter = 0
        self._last_frame_timestamp = None
        self._last_observation_timestamp = None
        self.tracker.reset()
        self.state_estimator.reset()
        self.health_monitor = HealthMonitor(self.config)
        self.timing_profiler = RollingTimingProfiler(window_size=50)

    def process_frame(
        self,
        raw_frame: Any,
        timestamp: float,
        ego_state: EgoMotionState | None = None,
        explicit_observations: Sequence[PhysicalObservation] | None = None,
    ) -> RuntimeResult:
        """Execute all pipeline stages for a single frame.

        Args:
            raw_frame: Sensory frame input (e.g. image array, synthetic entity tuple, or camera buffer).
            timestamp: Monotonic frame capture timestamp (seconds).
            ego_state: Optional ego-motion state. Defaults to stationary origin if None.
            explicit_observations: Optional pre-extracted PhysicalObservations (bypasses perception stage).

        Returns:
            RuntimeResult containing decisions, risks, active tracks, and timing breakdown.
        """
        self._frame_counter += 1
        frame_id = self._frame_counter
        t_total_start = time.perf_counter()

        context_builder = ContextBuilder(frame_id=frame_id, timestamp=timestamp, raw_frame=raw_frame)
        current_ego = (
            ego_state
            if ego_state is not None
            else EgoMotionState(position=Vector2(0.0, 0.0), velocity=Vector2(0.0, 0.0), timestamp=timestamp)
        )

        # Validate timestamp delta
        if self._last_frame_timestamp is not None:
            dt = timestamp - self._last_frame_timestamp
            if dt < 0.0:
                self.health_monitor.record_failure(f"Out of order timestamp {timestamp} < {self._last_frame_timestamp}")
                decision = create_fail_safe_decision(timestamp, DecisionReason.EMERGENCY_COLLISION_RISK, self.config.fail_safe_action)
                risk = create_fail_safe_risk_assessment(timestamp)
                total_duration = (time.perf_counter() - t_total_start) * 1000.0
                timing = StageTiming(total_ms=total_duration, effective_fps=1000.0 / max(0.001, total_duration))
                health = self.health_monitor.assess_health(0, 0, self.timing_profiler.average_latency_ms, self.timing_profiler.average_fps)
                return RuntimeResult(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    processing_duration_ms=total_duration,
                    decision=decision,
                    risk_assessment=risk,
                    active_track_count=0,
                    detection_count=0,
                    observation_count=0,
                    runtime_status=health.status,
                    timing=timing,
                    health=health,
                    context=context_builder.build(),
                )

        self._last_frame_timestamp = timestamp

        # Check for stale perception timeout
        is_stale = False
        if self._last_observation_timestamp is not None:
            if (timestamp - self._last_observation_timestamp) > self.config.stale_observation_timeout:
                is_stale = True

        # --- Stage 1: Perception ---
        t_perc_start = time.perf_counter()
        detections: list[Detection] = []
        observations: list[PhysicalObservation] = []
        rejections: list[DetectionRejection] = []

        if explicit_observations is not None:
            observations = list(explicit_observations)
            if observations:
                self._last_observation_timestamp = timestamp
                is_stale = False
        else:
            try:
                observations, rejections = self.perception_pipeline.process_frame(raw_frame, timestamp)
                if observations:
                    self._last_observation_timestamp = timestamp
                    is_stale = False
            except Exception as e:
                self.health_monitor.record_failure(f"Perception stage error: {e}")
                observations = []

        t_perc_end = time.perf_counter()
        perc_ms = (t_perc_end - t_perc_start) * 1000.0
        context_builder.set_perception_output(detections, observations, rejections)

        # --- Stage 2: Tracking & Association ---
        t_trk_start = time.perf_counter()
        tracked_observations: list[TrackedObservation] = []
        try:
            tracking_result = self.tracker.update(observations, timestamp)
            tracked_observations = list(tracking_result.tracked_observations)
        except Exception as e:
            self.health_monitor.record_failure(f"Tracking stage error: {e}")
        t_trk_end = time.perf_counter()
        trk_ms = (t_trk_end - t_trk_start) * 1000.0
        context_builder.set_tracking_output(tracked_observations)

        # --- Stage 3: State Estimation ---
        t_est_start = time.perf_counter()
        active_objects: dict[int, ObjectState] = {}
        stale_track_count = 0
        try:
            for trk in self.tracker.active_tracks:
                st = self.state_estimator.get_state(trk.track_id)
                if st is not None:
                    active_objects[trk.track_id] = st
                if trk.missed_frames > 0:
                    stale_track_count += 1
            world_state = WorldState(timestamp=timestamp, objects=active_objects)
        except Exception as e:
            self.health_monitor.record_failure(f"State estimation error: {e}")
            world_state = WorldState(timestamp=timestamp, objects={})
        t_est_end = time.perf_counter()
        est_ms = (t_est_end - t_est_start) * 1000.0
        context_builder.set_world_state(world_state)

        # --- Stage 4: Trajectory Prediction ---
        t_pred_start = time.perf_counter()
        trajectories: dict[int, Trajectory] = {}
        try:
            for obj_id, obj_state in world_state.objects.items():
                traj = self.predictor.predict(obj_state, horizon=self.config.prediction_horizon)
                trajectories[obj_id] = traj
        except Exception as e:
            self.health_monitor.record_failure(f"Prediction stage error: {e}")
        t_pred_end = time.perf_counter()
        pred_ms = (t_pred_end - t_pred_start) * 1000.0
        context_builder.set_trajectories(trajectories)

        # --- Stage 5: Risk Assessment ---
        t_risk_start = time.perf_counter()
        risk_assessment: RiskAssessment
        try:
            risk_assessment = self.risk_evaluator.get_primary_threat(world_state, ego=current_ego)
        except Exception as e:
            self.health_monitor.record_failure(f"Risk evaluation error: {e}")
            risk_assessment = create_fail_safe_risk_assessment(timestamp)
        t_risk_end = time.perf_counter()
        risk_ms = (t_risk_end - t_risk_start) * 1000.0
        context_builder.set_risk(risk_assessment)

        # --- Stage 6: Decision Engine ---
        t_dec_start = time.perf_counter()
        decision: NavigationDecision
        try:
            decision_ctx = DecisionContext(
                ego_position=current_ego.position,
                risk_assessment=risk_assessment,
                predicted_obstacles=tuple(world_state.objects.values()),
                timestamp=timestamp,
            )
            decision = self.decision_engine.decide_navigation(decision_ctx)

            # If perception is stale, enforce configured conservative policy
            if is_stale and decision.action in (NavigationAction.CONTINUE,):
                decision = create_fail_safe_decision(
                    timestamp=timestamp,
                    reason=DecisionReason.PATH_BLOCKED,
                    action=NavigationAction.CAUTION,
                )
        except Exception as e:
            self.health_monitor.record_failure(f"Decision stage error: {e}")
            decision = create_fail_safe_decision(timestamp, DecisionReason.EMERGENCY_COLLISION_RISK, self.config.fail_safe_action)
        t_dec_end = time.perf_counter()
        dec_ms = (t_dec_end - t_dec_start) * 1000.0
        context_builder.set_decision(decision)

        # Total timing computation
        t_total_end = time.perf_counter()
        total_ms = (t_total_end - t_total_start) * 1000.0
        fps = 1000.0 / max(0.001, total_ms)

        timing = StageTiming(
            perception_ms=perc_ms,
            tracking_ms=trk_ms,
            estimation_ms=est_ms,
            prediction_ms=pred_ms,
            risk_ms=risk_ms,
            decision_ms=dec_ms,
            total_ms=total_ms,
            effective_fps=fps,
        )
        context_builder.set_timing(timing)
        self.timing_profiler.record(timing)

        # Update health monitor
        self.health_monitor.record_success()
        health = self.health_monitor.assess_health(
            active_tracks=len(world_state.objects),
            stale_tracks=stale_track_count,
            average_latency_ms=self.timing_profiler.average_latency_ms,
            current_fps=self.timing_profiler.average_fps,
        )

        return RuntimeResult(
            frame_id=frame_id,
            timestamp=timestamp,
            processing_duration_ms=total_ms,
            decision=decision,
            risk_assessment=risk_assessment,
            active_track_count=len(world_state.objects),
            detection_count=len(observations),
            observation_count=len(observations),
            runtime_status=health.status,
            timing=timing,
            health=health,
            context=context_builder.build(),
        )
