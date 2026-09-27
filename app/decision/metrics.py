"""Quantitative decision quality and avoidance performance metrics."""

from __future__ import annotations

import math
from typing import Sequence

from app.decision.model import NavigationAction, NavigationDecision


def decision_accuracy(
    predicted_actions: Sequence[NavigationAction],
    reference_actions: Sequence[NavigationAction],
) -> float:
    """Compute decision accuracy against an explicit reference policy oracle.

    Args:
        predicted_actions: Sequence of predicted NavigationAction decisions.
        reference_actions: Sequence of expected NavigationAction oracle decisions.

    Returns:
        Percentage of matching decisions in [0.0, 100.0].

    Raises:
        ValueError: If sequences are empty or have unequal lengths.
    """
    n = len(predicted_actions)
    if n == 0:
        raise ValueError("Cannot compute decision accuracy on empty sequence.")
    if n != len(reference_actions):
        raise ValueError(f"Sequence length mismatch: {n} predicted vs {len(reference_actions)} reference.")

    matches = sum(1 for p, r in zip(predicted_actions, reference_actions) if p == r)
    return (matches / n) * 100.0


def unnecessary_stop_rate(
    predicted_actions: Sequence[NavigationAction],
    reference_actions: Sequence[NavigationAction],
) -> float:
    """Compute rate of false stops on safe or non-hazardous situations.

    Measures fraction of times the system stops when the reference policy allows motion (CONTINUE / CAUTION).

    Returns:
        Percentage in [0.0, 100.0].
    """
    safe_refs = [
        (p, r) for p, r in zip(predicted_actions, reference_actions)
        if r in (NavigationAction.CONTINUE, NavigationAction.CAUTION)
    ]
    if not safe_refs:
        return 0.0

    false_stops = sum(
        1 for p, _ in safe_refs
        if p in (NavigationAction.STOP, NavigationAction.EMERGENCY_STOP)
    )
    return (false_stops / len(safe_refs)) * 100.0


def collision_avoidance_rate(
    decisions: Sequence[NavigationDecision],
    ground_truth_hazard_occurred: bool,
) -> float:
    """Compute the fraction of hazardous situations where the decision engine intervened with an evasive or stopping action.

    Intervention actions: STOP, EMERGENCY_STOP, TURN_LEFT, TURN_RIGHT, SLOW_DOWN.

    Returns:
        Percentage in [0.0, 100.0].
    """
    if not decisions:
        return 0.0
    if not ground_truth_hazard_occurred:
        return 100.0

    interventions = sum(
        1 for d in decisions
        if d.action in (
            NavigationAction.EMERGENCY_STOP,
            NavigationAction.STOP,
            NavigationAction.TURN_LEFT,
            NavigationAction.TURN_RIGHT,
            NavigationAction.SLOW_DOWN,
        )
    )
    return (interventions / len(decisions)) * 100.0


def mean_heading_deviation(decisions: Sequence[NavigationDecision]) -> float:
    """Compute average absolute recommended turn angle (radians) across active turning decisions."""
    turn_angles = [
        abs(d.recommended_heading)
        for d in decisions
        if d.recommended_heading is not None and d.action in (NavigationAction.TURN_LEFT, NavigationAction.TURN_RIGHT)
    ]
    if not turn_angles:
        return 0.0
    return sum(turn_angles) / len(turn_angles)


def decision_switch_rate(
    decisions: Sequence[NavigationDecision],
    duration: float,
) -> float:
    """Compute decision switching frequency (action changes per second).

    Lower switch rate indicates greater temporal stability / reduced oscillation.

    Returns:
        Switches per second.
    """
    if len(decisions) <= 1 or duration <= 0.0:
        return 0.0

    switches = 0
    for i in range(1, len(decisions)):
        if decisions[i].action != decisions[i - 1].action:
            switches += 1

    return switches / duration
