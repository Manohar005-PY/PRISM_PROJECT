"""Warning lead time and alert timeliness evaluation engine for PRISM Part 17."""

from __future__ import annotations

from typing import Any, Sequence

from simulation.evaluation.alignment import AlignedFrame
from simulation.evaluation.models import MetricStats
from simulation.evaluation.state_metrics import compute_metric_stats


WARNING_ACTIONS = ("SLOW_DOWN", "TURN_LEFT", "TURN_RIGHT", "STOP", "EMERGENCY_STOP")
WARNING_RISK_LEVELS = ("HIGH", "CRITICAL")


class WarningEvaluator:
    """Evaluates warning lead time before collisions and false warning rates on safe runs."""

    def evaluate_run(self, aligned_frames: Sequence[AlignedFrame]) -> dict[str, Any]:
        """Compute warning lead time, maintained alert duration, and false alarm metrics for a run."""
        if not aligned_frames:
            return {
                "has_collision_event": False,
                "warning_issued": False,
                "first_warning_time": None,
                "collision_time": None,
                "lead_time_s": None,
                "warning_maintained_ratio": None,
                "false_warning_detected": False,
                "time_in_warning_state_s": 0.0,
            }

        first_collision_time: float | None = None
        first_warning_time: float | None = None
        warning_frames_count = 0
        total_frames = len(aligned_frames)
        dt = aligned_frames[1].timestamp - aligned_frames[0].timestamp if total_frames > 1 else 0.05

        for frame in aligned_frames:
            t = frame.timestamp
            gt_col = frame.gt_collision_state
            if gt_col.get("has_collided", False) and first_collision_time is None:
                first_collision_time = t

            pr_act = frame.prism_output.get("decision_action", "CONTINUE")
            pr_risk = frame.prism_output.get("risk_level", "SAFE")

            is_warning = (pr_act in WARNING_ACTIONS or pr_risk in WARNING_RISK_LEVELS)
            if is_warning:
                warning_frames_count += 1
                if first_warning_time is None:
                    first_warning_time = t

        has_collision = (first_collision_time is not None)
        lead_time_s: float | None = None
        warning_maintained_ratio: float | None = None
        false_warning_detected = False

        if has_collision:
            # Genuine collision run
            if first_warning_time is not None and first_warning_time <= first_collision_time:
                lead_time_s = round(first_collision_time - first_warning_time, 4)

                # Check if warning was maintained between first warning and collision
                f_start = int(round(first_warning_time / dt))
                f_end = int(round(first_collision_time / dt))
                span = max(1, f_end - f_start + 1)
                active_in_span = sum(
                    1 for k in range(f_start, min(total_frames, f_end + 1))
                    if (aligned_frames[k].prism_output.get("decision_action") in WARNING_ACTIONS
                        or aligned_frames[k].prism_output.get("risk_level") in WARNING_RISK_LEVELS)
                )
                warning_maintained_ratio = round(active_in_span / span, 4)
            else:
                # Missed warning or warning arrived after collision occurred
                lead_time_s = None
        else:
            # Safe / Near-miss run: evaluate false alarms
            if first_warning_time is not None:
                # Triggered HIGH/CRITICAL warning on a safe run
                false_warning_detected = True

        time_in_warning = round(warning_frames_count * dt, 4)

        return {
            "has_collision_event": has_collision,
            "warning_issued": (first_warning_time is not None),
            "first_warning_time": first_warning_time,
            "collision_time": first_collision_time,
            "lead_time_s": lead_time_s,
            "warning_maintained_ratio": warning_maintained_ratio,
            "false_warning_detected": false_warning_detected,
            "time_in_warning_state_s": time_in_warning,
        }
