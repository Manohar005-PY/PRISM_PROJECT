"""Scientific evaluation metrics: Trajectory RMSE, horizon error, CPA/TTC error, hazard detection rate, and decision switch rate."""

from __future__ import annotations

import math
from typing import Sequence

from app.core.enums import RiskLevel
from app.decision.model import NavigationAction
from evaluation.models import EvaluationSample


def compute_hazard_detection_rate(samples: Sequence[EvaluationSample]) -> float:
    """Calculate the ratio of true hazard frames correctly classified as MEDIUM/HIGH/CRITICAL risk or alert action."""
    hazard_samples = [s for s in samples if s.is_scene_hazard]
    if not hazard_samples:
        return 100.0

    detected = 0
    for s in hazard_samples:
        is_alert_risk = s.risk.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
        is_action_taken = s.decision.action in (
            NavigationAction.STOP,
            NavigationAction.EMERGENCY_STOP,
            NavigationAction.TURN_LEFT,
            NavigationAction.TURN_RIGHT,
            NavigationAction.SLOW_DOWN,
            NavigationAction.CAUTION,
        )
        if is_alert_risk or is_action_taken:
            detected += 1

    return (detected / len(hazard_samples)) * 100.0


def compute_false_alarm_rate(samples: Sequence[EvaluationSample]) -> float:
    """Calculate the ratio of safe frames incorrectly classified as high hazard or emergency action."""
    safe_samples = [s for s in samples if not s.is_scene_hazard]
    if not safe_samples:
        return 0.0

    false_alarms = 0
    for s in safe_samples:
        is_critical_risk = s.risk.risk_level in (RiskLevel.CRITICAL,)
        is_emergency_stop = s.decision.action in (NavigationAction.EMERGENCY_STOP, NavigationAction.STOP)
        if is_critical_risk or is_emergency_stop:
            false_alarms += 1

    return (false_alarms / len(safe_samples)) * 100.0


def compute_unnecessary_stop_rate(samples: Sequence[EvaluationSample]) -> float:
    """Calculate ratio of frames commanding STOP when scene is physically safe."""
    safe_samples = [s for s in samples if not s.is_scene_hazard]
    if not safe_samples:
        return 0.0

    stops = sum(1 for s in safe_samples if s.decision.action in (NavigationAction.STOP, NavigationAction.EMERGENCY_STOP))
    return (stops / len(safe_samples)) * 100.0


def compute_reaction_time(samples: Sequence[EvaluationSample]) -> float | None:
    """Calculate time in seconds from first true hazard onset until the first intervention action."""
    hazard_onset_time: float | None = None
    intervention_time: float | None = None

    for s in samples:
        if s.is_scene_hazard and hazard_onset_time is None:
            hazard_onset_time = s.timestamp

        if hazard_onset_time is not None and intervention_time is None:
            if s.decision.action in (
                NavigationAction.STOP,
                NavigationAction.EMERGENCY_STOP,
                NavigationAction.SLOW_DOWN,
                NavigationAction.TURN_LEFT,
                NavigationAction.TURN_RIGHT,
                NavigationAction.CAUTION,
            ):
                intervention_time = s.timestamp

    if hazard_onset_time is not None and intervention_time is not None:
        return max(0.0, intervention_time - hazard_onset_time)
    return None


def compute_decision_switch_rate(samples: Sequence[EvaluationSample]) -> float:
    """Calculate rate of navigation action switches per second (measures policy stability/hysteresis)."""
    if len(samples) < 2:
        return 0.0

    switches = 0
    prev_action = samples[0].decision.action
    for s in samples[1:]:
        if s.decision.action != prev_action:
            switches += 1
            prev_action = s.decision.action

    duration = samples[-1].timestamp - samples[0].timestamp
    if duration <= 0.0:
        return 0.0

    return switches / duration


def compute_position_rmse(samples: Sequence[EvaluationSample]) -> float:
    """Calculate root mean square position error across all tracked obstacles and predictions."""
    errors_sq: list[float] = []

    for s in samples:
        for obj_id, pred in s.predictions.items():
            if obj_id in s.ground_truth:
                gt = s.ground_truth[obj_id]
                dx = pred.predicted_position.x - gt.position.x
                dz = pred.predicted_position.z - gt.position.z
                errors_sq.append(dx**2 + dz**2)

    if not errors_sq:
        return 0.0
    return math.sqrt(sum(errors_sq) / len(errors_sq))
