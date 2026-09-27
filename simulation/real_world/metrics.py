"""Quantitative metrics evaluation for PRISM Part 20 Real-World Benchmark."""

from __future__ import annotations

from typing import Sequence

from simulation.real_world.models import (
    RealWorldFrameResult,
    RealWorldRunMetadata,
    RealWorldRunResult,
)

EMERGENCY_ACTIONS = ("STOP", "EMERGENCY_STOP")
EMERGENCY_RISK_LEVELS = ("CRITICAL",)
CAUTION_ACTIONS = ("SLOW_DOWN", "TURN_LEFT", "TURN_RIGHT")
CAUTION_RISK_LEVELS = ("MODERATE", "HIGH")
ALL_WARNING_ACTIONS = ("SLOW_DOWN", "TURN_LEFT", "TURN_RIGHT", "STOP", "EMERGENCY_STOP")
ALL_WARNING_RISK_LEVELS = ("HIGH", "CRITICAL")
WARNING_ACTIONS = ALL_WARNING_ACTIONS
WARNING_RISK_LEVELS = ALL_WARNING_RISK_LEVELS


class RealWorldMetricsEvaluator:
    """Computes quantitative real-world warning, tracking, and control stability metrics."""

    def __init__(self) -> None:
        pass

    def evaluate_run(
        self,
        metadata: RealWorldRunMetadata,
        frame_results: Sequence[RealWorldFrameResult],
    ) -> RealWorldRunResult:
        """Compute summary benchmark metrics for a single video run."""
        total_frames = len(frame_results)
        duration_s = metadata.duration_s if metadata.duration_s > 0.0 else (total_frames / metadata.fps if metadata.fps > 0.0 else 0.0)

        first_warn_time: float | None = None
        first_caution_time: float | None = None
        first_emergency_time: float | None = None

        actions: list[str] = []
        latencies: list[float] = []
        det_counts: list[int] = []
        frames_with_detections = 0
        frames_with_tracks = 0
        frames_with_det_and_tracks = 0
        zero_detection_frames = 0
        coasting_frames = 0

        for f in frame_results:
            actions.append(f.decision_action)
            latencies.append(f.latency_ms)
            det_counts.append(f.raw_detections_count)

            if f.raw_detections_count > 0:
                frames_with_detections += 1
            else:
                zero_detection_frames += 1

            if f.active_tracks_count > 0:
                frames_with_tracks += 1

            if f.raw_detections_count > 0 and f.active_tracks_count > 0:
                frames_with_det_and_tracks += 1
            elif f.raw_detections_count == 0 and f.active_tracks_count > 0:
                coasting_frames += 1

            is_warn = (f.decision_action in ALL_WARNING_ACTIONS or f.risk_level in ALL_WARNING_RISK_LEVELS)
            if is_warn and first_warn_time is None:
                first_warn_time = f.timestamp

            is_caution = (f.decision_action in CAUTION_ACTIONS or f.risk_level in CAUTION_RISK_LEVELS)
            if is_caution and first_caution_time is None:
                first_caution_time = f.timestamp

            is_emergency = (f.decision_action in EMERGENCY_ACTIONS or f.risk_level in EMERGENCY_RISK_LEVELS)
            if is_emergency and first_emergency_time is None:
                first_emergency_time = f.timestamp

        # Warning lead time and semantic classification
        t_coll = metadata.collision_timestamp_s
        lead_time_s: float | None = None

        if metadata.has_collision:
            warned = (first_warn_time is not None and (t_coll is None or first_warn_time <= t_coll))
            collision_recalled = warned
            collision_missed = not warned
            caution_issued = (first_caution_time is not None and (t_coll is None or first_caution_time <= t_coll))
            emergency_issued = (first_emergency_time is not None and (t_coll is None or first_emergency_time <= t_coll))
            proximity_caution_alert = False
            false_collision_warning = False
            false_warn = False
            if t_coll is not None and first_warn_time is not None and first_warn_time <= t_coll:
                lead_time_s = round(t_coll - first_warn_time, 4)
        else:
            warned = (first_warn_time is not None)
            collision_recalled = False
            collision_missed = False
            caution_issued = (first_caution_time is not None)
            emergency_issued = (first_emergency_time is not None)
            proximity_caution_alert = caution_issued
            false_collision_warning = emergency_issued
            false_warn = false_collision_warning

        # Decision changes and steering reversals
        changes = 0
        reversals = 0
        prev_act: str | None = None
        prev_turn: str | None = None
        for a in actions:
            if prev_act is None:
                prev_act = a
            elif a != prev_act:
                changes += 1
                prev_act = a
            if a in ("TURN_LEFT", "TURN_RIGHT"):
                if prev_turn is not None and a != prev_turn:
                    reversals += 1
                prev_turn = a

        mean_det = (sum(det_counts) / len(det_counts)) if len(det_counts) > 0 else 0.0
        mean_lat = (sum(latencies) / len(latencies)) if len(latencies) > 0 else 0.0

        # Scientific ratio computation with explicit undefined (None) states
        track_coverage = (frames_with_tracks / total_frames) if total_frames > 0 else None
        det_assoc_coverage = (frames_with_det_and_tracks / frames_with_detections) if frames_with_detections > 0 else None
        gap_bridging = (coasting_frames / zero_detection_frames) if zero_detection_frames > 0 else None
        track_continuity = det_assoc_coverage

        return RealWorldRunResult(
            run_id=metadata.run_id,
            scenario_family=metadata.scenario_family,
            video_path=metadata.video_path,
            has_collision=metadata.has_collision,
            collision_timestamp_s=t_coll,
            first_warning_timestamp_s=first_warn_time,
            first_caution_timestamp_s=first_caution_time,
            first_emergency_timestamp_s=first_emergency_time,
            warning_lead_time_s=lead_time_s,
            warned=warned,
            collision_recalled=collision_recalled,
            collision_missed=collision_missed,
            caution_alert_issued=caution_issued,
            emergency_warning_issued=emergency_issued,
            proximity_caution_alert=proximity_caution_alert,
            false_collision_warning=false_collision_warning,
            false_warning=false_warn,
            total_frames=total_frames,
            duration_s=duration_s,
            fps=metadata.fps,
            mean_detection_count=mean_det,
            tracking_continuity_ratio=track_continuity,
            mean_latency_ms=mean_lat,
            decision_changes=changes,
            steering_reversals=reversals,
            track_coverage_ratio=track_coverage,
            detection_association_coverage_ratio=det_assoc_coverage,
            zero_detection_frames=zero_detection_frames,
            coasting_frames=coasting_frames,
            gap_bridging_rate=gap_bridging,
            frame_results=list(frame_results),
        )
