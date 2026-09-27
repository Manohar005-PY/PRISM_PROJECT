"""Deterministic priority-ordered safety rule evaluation for PRISM decision making."""

from __future__ import annotations

from app.core.enums import RiskLevel
from app.decision.context import DecisionContext
from app.decision.heading import SafeHeading
from app.decision.model import DecisionReason, NavigationAction


def evaluate_safety_rules(
    risk_level: RiskLevel,
    best_heading: SafeHeading | None,
    context: DecisionContext,
) -> tuple[NavigationAction, DecisionReason]:
    """Evaluate deterministic safety rules in explicit priority order.

    Precedence: CRITICAL -> HIGH -> MEDIUM -> LOW -> SAFE.

    Args:
        risk_level: Assessed hazard level from Part 6.
        best_heading: Optimal safe heading calculated by geometric clearance analysis (or None).
        context: DecisionContext containing user kinematics and constraints.

    Returns:
        (NavigationAction, DecisionReason) tuple.
    """
    # ------------------------------------------------------------------------
    # RULE 1: CRITICAL RISK
    # ------------------------------------------------------------------------
    if risk_level == RiskLevel.CRITICAL:
        ttc = context.risk_assessment.ttc if context.risk_assessment is not None else None
        if ttc is not None and ttc <= 0.8:
            return (NavigationAction.EMERGENCY_STOP, DecisionReason.EMERGENCY_COLLISION_RISK)
        return (NavigationAction.STOP, DecisionReason.CRITICAL_COLLISION_RISK)

    # ------------------------------------------------------------------------
    # RULE 2: HIGH RISK
    # ------------------------------------------------------------------------
    if risk_level == RiskLevel.HIGH:
        if best_heading is not None:
            # Evasive turn when deviation is meaningful
            if best_heading.angle < -0.05:
                return (NavigationAction.TURN_LEFT, DecisionReason.SAFE_PATH_AVAILABLE)
            elif best_heading.angle > 0.05:
                return (NavigationAction.TURN_RIGHT, DecisionReason.SAFE_PATH_AVAILABLE)
            elif best_heading.clearance >= 2.0:
                return (NavigationAction.SLOW_DOWN, DecisionReason.SAFE_PATH_AVAILABLE)

        # No safe alternative corridor exists
        return (NavigationAction.STOP, DecisionReason.NO_SAFE_PATH)

    # ------------------------------------------------------------------------
    # RULE 3: MEDIUM RISK
    # ------------------------------------------------------------------------
    if risk_level == RiskLevel.MEDIUM:
        if best_heading is not None:
            if best_heading.angle < -0.10:
                return (NavigationAction.TURN_LEFT, DecisionReason.SAFE_PATH_AVAILABLE)
            elif best_heading.angle > 0.10:
                return (NavigationAction.TURN_RIGHT, DecisionReason.SAFE_PATH_AVAILABLE)
            return (NavigationAction.SLOW_DOWN, DecisionReason.MEDIUM_COLLISION_RISK)

        return (NavigationAction.STOP, DecisionReason.PATH_BLOCKED)

    # ------------------------------------------------------------------------
    # RULE 4: LOW RISK
    # ------------------------------------------------------------------------
    if risk_level == RiskLevel.LOW:
        if best_heading is not None and best_heading.clearance < 2.5:
            return (NavigationAction.CAUTION, DecisionReason.LOW_COLLISION_RISK)
        return (NavigationAction.CONTINUE, DecisionReason.NO_HAZARD)

    # ------------------------------------------------------------------------
    # RULE 5: SAFE (NO HAZARD)
    # ------------------------------------------------------------------------
    return (NavigationAction.CONTINUE, DecisionReason.NO_HAZARD)
