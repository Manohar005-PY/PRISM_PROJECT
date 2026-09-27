"""Local motion candidate generation, hard safety constraint filtering, and cost ranking."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from app.core.enums import RiskLevel
from app.decision.context import DecisionContext
from app.decision.heading import SafeHeading, compute_safe_headings, select_best_heading
from app.decision.model import DecisionReason, NavigationAction


@dataclass(frozen=True)
class CandidateAction:
    """Evaluated local motion candidate action with deterministic cost.

    Attributes:
        action: Proposed NavigationAction.
        cost: Multi-component soft cost (lower = more preferable).
        recommended_heading: Proposed ground-plane angle in radians (or None).
        reason: Justifying DecisionReason.
        is_safe: True iff candidate satisfies all hard geometric safety constraints.
    """

    action: NavigationAction
    cost: float
    recommended_heading: float | None
    reason: DecisionReason
    is_safe: bool


class LocalMotionPlanner:
    """Evaluates local motion alternatives with hard geometric safety bounds and soft cost ranking."""

    def __init__(
        self,
        weight_risk: float = 0.40,
        weight_heading: float = 0.30,
        weight_clearance: float = 0.20,
        weight_speed: float = 0.10,
    ) -> None:
        self.w_risk = weight_risk
        self.w_heading = weight_heading
        self.w_clearance = weight_clearance
        self.w_speed = weight_speed

    def plan_candidates(
        self,
        context: DecisionContext,
        safe_headings: Sequence[SafeHeading],
    ) -> list[CandidateAction]:
        """Generate and evaluate candidate navigation actions against safety constraints and cost functions.

        Args:
            context: Current DecisionContext.
            safe_headings: List of sampled SafeHeading instances from geometric ray casting.

        Returns:
            List of CandidateAction instances ranked by cost.
        """
        risk_level = context.risk_assessment.risk_level if context.risk_assessment is not None else RiskLevel.SAFE
        risk_score = (context.risk_assessment.risk_score / 100.0) if context.risk_assessment is not None else 0.0

        # Find forward, best left, and best right headings
        forward_heading = next((h for h in safe_headings if abs(h.angle) <= 0.05), None)
        left_headings = [h for h in safe_headings if h.angle < -0.05 and h.is_valid]
        right_headings = [h for h in safe_headings if h.angle > 0.05 and h.is_valid]

        best_left = min(left_headings, key=lambda h: (abs(h.angle - context.preferred_heading), -h.clearance)) if left_headings else None
        best_right = min(right_headings, key=lambda h: (abs(h.angle - context.preferred_heading), -h.clearance)) if right_headings else None

        candidates: list[CandidateAction] = []

        # --------------------------------------------------------------------
        # Candidate 1: CONTINUE
        # --------------------------------------------------------------------
        forward_safe = forward_heading is not None and forward_heading.is_valid and risk_level not in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        if forward_safe and forward_heading is not None:
            c_clear = 1.0 - min(1.0, forward_heading.clearance / 6.0)
            cost_continue = (
                self.w_risk * (risk_score if risk_level == RiskLevel.LOW else 0.0)
                + self.w_heading * (abs(forward_heading.angle - context.preferred_heading) / context.max_turn_angle)
                + self.w_clearance * c_clear
                + self.w_speed * 0.0
            )
            candidates.append(
                CandidateAction(
                    action=NavigationAction.CONTINUE,
                    cost=cost_continue,
                    recommended_heading=0.0,
                    reason=DecisionReason.NO_HAZARD if risk_level == RiskLevel.SAFE else DecisionReason.LOW_COLLISION_RISK,
                    is_safe=True,
                )
            )

        # --------------------------------------------------------------------
        # Candidate 2: SLOW_DOWN
        # --------------------------------------------------------------------
        slow_safe = forward_heading is not None and forward_heading.clearance >= 1.5 and risk_level != RiskLevel.CRITICAL
        if slow_safe and forward_heading is not None:
            c_clear = 1.0 - min(1.0, forward_heading.clearance / 6.0)
            cost_slow = (
                self.w_risk * (risk_score * 0.5)
                + self.w_heading * (abs(forward_heading.angle - context.preferred_heading) / context.max_turn_angle)
                + self.w_clearance * c_clear
                + self.w_speed * 0.25
                + 0.10  # Moderate maneuver penalty
            )
            candidates.append(
                CandidateAction(
                    action=NavigationAction.SLOW_DOWN,
                    cost=cost_slow,
                    recommended_heading=0.0,
                    reason=DecisionReason.MEDIUM_COLLISION_RISK if risk_level == RiskLevel.MEDIUM else DecisionReason.SAFE_PATH_AVAILABLE,
                    is_safe=True,
                )
            )

        # --------------------------------------------------------------------
        # Candidate 3: TURN_LEFT
        # --------------------------------------------------------------------
        if best_left is not None and risk_level != RiskLevel.CRITICAL:
            c_clear = 1.0 - min(1.0, best_left.clearance / 6.0)
            # Turning penalty prevents unnecessary turning when forward is clear
            turn_penalty = 0.0 if (risk_level in (RiskLevel.HIGH, RiskLevel.MEDIUM) or not forward_safe) else 0.40
            cost_left = (
                self.w_risk * (risk_score * 0.4)
                + self.w_heading * (abs(best_left.angle - context.preferred_heading) / context.max_turn_angle)
                + self.w_clearance * c_clear
                + self.w_speed * 0.15
                + turn_penalty
            )
            candidates.append(
                CandidateAction(
                    action=NavigationAction.TURN_LEFT,
                    cost=cost_left,
                    recommended_heading=best_left.angle,
                    reason=DecisionReason.SAFE_PATH_AVAILABLE,
                    is_safe=True,
                )
            )

        # --------------------------------------------------------------------
        # Candidate 4: TURN_RIGHT
        # --------------------------------------------------------------------
        if best_right is not None and risk_level != RiskLevel.CRITICAL:
            c_clear = 1.0 - min(1.0, best_right.clearance / 6.0)
            turn_penalty = 0.0 if (risk_level in (RiskLevel.HIGH, RiskLevel.MEDIUM) or not forward_safe) else 0.40
            cost_right = (
                self.w_risk * (risk_score * 0.4)
                + self.w_heading * (abs(best_right.angle - context.preferred_heading) / context.max_turn_angle)
                + self.w_clearance * c_clear
                + self.w_speed * 0.15
                + turn_penalty
            )
            candidates.append(
                CandidateAction(
                    action=NavigationAction.TURN_RIGHT,
                    cost=cost_right,
                    recommended_heading=best_right.angle,
                    reason=DecisionReason.SAFE_PATH_AVAILABLE,
                    is_safe=True,
                )
            )

        # --------------------------------------------------------------------
        # Candidate 5: CAUTION
        # --------------------------------------------------------------------
        if risk_level == RiskLevel.LOW and forward_heading is not None and forward_heading.is_valid:
            cost_caution = (
                self.w_risk * risk_score
                + self.w_heading * (abs(forward_heading.angle - context.preferred_heading) / context.max_turn_angle)
                + self.w_clearance * 0.3
                + self.w_speed * 0.10
            )
            candidates.append(
                CandidateAction(
                    action=NavigationAction.CAUTION,
                    cost=cost_caution,
                    recommended_heading=forward_heading.angle,
                    reason=DecisionReason.LOW_COLLISION_RISK,
                    is_safe=True,
                )
            )

        # --------------------------------------------------------------------
        # Candidate 6: STOP & EMERGENCY_STOP (Always geometrically safe)
        # --------------------------------------------------------------------
        ttc = context.risk_assessment.ttc if context.risk_assessment is not None else None
        is_emergency = (risk_level == RiskLevel.CRITICAL) and (ttc is not None and ttc <= 0.8)

        if is_emergency:
            candidates.append(
                CandidateAction(
                    action=NavigationAction.EMERGENCY_STOP,
                    cost=0.05,  # Top priority for emergency
                    recommended_heading=None,
                    reason=DecisionReason.EMERGENCY_COLLISION_RISK,
                    is_safe=True,
                )
            )
        else:
            has_safe_turn = (best_left is not None) or (best_right is not None)
            if risk_level == RiskLevel.CRITICAL:
                cost_stop = 0.10
            elif risk_level == RiskLevel.HIGH and not has_safe_turn:
                cost_stop = 0.20
            elif risk_level == RiskLevel.HIGH and has_safe_turn:
                cost_stop = 0.55  # Prefer safe evasive turn over stop when safe turn exists
            else:
                cost_stop = 0.80

            reason_stop = (
                DecisionReason.CRITICAL_COLLISION_RISK if risk_level == RiskLevel.CRITICAL
                else DecisionReason.NO_SAFE_PATH if not (has_safe_turn or forward_safe)
                else DecisionReason.HIGH_COLLISION_RISK
            )
            candidates.append(
                CandidateAction(
                    action=NavigationAction.STOP,
                    cost=cost_stop,
                    recommended_heading=None,
                    reason=reason_stop,
                    is_safe=True,
                )
            )

        # Sort candidates deterministically by cost
        candidates.sort(key=lambda c: (c.cost, -get_priority_score(c.action)))
        return candidates


def get_priority_score(action: NavigationAction) -> int:
    """Helper priority mapping for tie-breaking."""
    order = {
        NavigationAction.EMERGENCY_STOP: 6,
        NavigationAction.STOP: 5,
        NavigationAction.TURN_LEFT: 4,
        NavigationAction.TURN_RIGHT: 4,
        NavigationAction.SLOW_DOWN: 3,
        NavigationAction.CAUTION: 2,
        NavigationAction.CONTINUE: 1,
    }
    return order.get(action, 0)
