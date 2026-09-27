"""Decision arbitration, lateral hysteresis memory, and emergency safety escalation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.core.enums import RiskLevel
from app.decision.context import DecisionContext
from app.decision.model import (
    DecisionReason,
    NavigationAction,
    NavigationDecision,
    get_action_priority,
)
from app.decision.planner import CandidateAction


@dataclass
class DecisionMemory:
    """Historical state memory for lateral hysteresis and oscillation prevention."""

    previous_action: NavigationAction | None = None
    previous_heading: float | None = None
    previous_risk_level: RiskLevel | None = None
    previous_timestamp: float = 0.0

    def update(self, decision: NavigationDecision) -> None:
        """Store the accepted decision into memory."""
        self.previous_action = decision.action
        self.previous_heading = decision.recommended_heading
        self.previous_risk_level = decision.risk_level
        self.previous_timestamp = decision.timestamp

    def reset(self) -> None:
        """Clear historical memory."""
        self.previous_action = None
        self.previous_heading = None
        self.previous_risk_level = None
        self.previous_timestamp = 0.0


class DecisionArbitrator:
    """Arbitrates candidate actions, enforces safety priority, and applies hysteresis."""

    def __init__(self, hysteresis_margin: float = 0.15) -> None:
        self.hysteresis_margin = hysteresis_margin

    def arbitrate(
        self,
        candidates: Sequence[CandidateAction],
        rules_action: NavigationAction,
        rules_reason: DecisionReason,
        context: DecisionContext,
        memory: DecisionMemory,
    ) -> NavigationDecision:
        """Select the final authoritative NavigationDecision.

        Args:
            candidates: Ranked list of safe CandidateAction instances from planner.
            rules_action: Top-priority action mandated by deterministic safety rules.
            rules_reason: Justifying reason from safety rules.
            context: Current DecisionContext.
            memory: DecisionMemory tracking previous frame decisions.

        Returns:
            Authoritative NavigationDecision instance.
        """
        risk_level = context.risk_assessment.risk_level if context.risk_assessment is not None else RiskLevel.SAFE
        primary_threat = (
            getattr(context.risk_assessment, "primary_threat_id", None)
            or getattr(context.risk_assessment, "object_id", None)
            if context.risk_assessment is not None else None
        )
        confidence = context.risk_assessment.confidence if context.risk_assessment is not None else 1.0
        t = context.timestamp

        # --------------------------------------------------------------------
        # 1. EMERGENCY & STOP ESCALATIONS BYPASS HYSTERESIS IMMEDIATELY
        # --------------------------------------------------------------------
        if rules_action in (NavigationAction.EMERGENCY_STOP, NavigationAction.STOP):
            cand = next((c for c in candidates if c.action == rules_action), None)
            reason = cand.reason if cand is not None else rules_reason
            return NavigationDecision(
                action=rules_action,
                reason=reason,
                confidence=confidence,
                primary_threat_id=primary_threat,
                recommended_heading=None,
                risk_level=risk_level,
                timestamp=t,
            )

        # --------------------------------------------------------------------
        # 2. SELECT TOP RANKED SAFE CANDIDATE
        # --------------------------------------------------------------------
        if not candidates:
            return NavigationDecision(
                action=NavigationAction.STOP,
                reason=DecisionReason.NO_SAFE_PATH,
                confidence=confidence,
                primary_threat_id=primary_threat,
                recommended_heading=None,
                risk_level=risk_level,
                timestamp=t,
            )

        chosen = candidates[0]

        # --------------------------------------------------------------------
        # 3. LATERAL HYSTERESIS FILTERING (TURN_LEFT <-> TURN_RIGHT)
        # --------------------------------------------------------------------
        held_by_hysteresis = False
        prev_act = memory.previous_action
        if prev_act in (NavigationAction.TURN_LEFT, NavigationAction.TURN_RIGHT):
            if chosen.action in (NavigationAction.TURN_LEFT, NavigationAction.TURN_RIGHT) and chosen.action != prev_act:
                prev_cand = next((c for c in candidates if c.action == prev_act), None)
                if prev_cand is not None:
                    cost_diff = prev_cand.cost - chosen.cost
                    if cost_diff < self.hysteresis_margin:
                        # Retain previous turn direction to prevent high-frequency jitter
                        chosen = CandidateAction(
                            action=prev_cand.action,
                            cost=prev_cand.cost,
                            recommended_heading=prev_cand.recommended_heading,
                            reason=DecisionReason.HYSTERESIS_HOLD,
                            is_safe=prev_cand.is_safe,
                        )
                        held_by_hysteresis = True

        # --------------------------------------------------------------------
        # 4. RULE PRIORITY ARBITRATION (IF NOT HELD BY HYSTERESIS)
        # --------------------------------------------------------------------
        if not held_by_hysteresis:
            if rules_action in (NavigationAction.TURN_LEFT, NavigationAction.TURN_RIGHT, NavigationAction.SLOW_DOWN, NavigationAction.CONTINUE):
                rule_cand = next((c for c in candidates if c.action == rules_action), None)
                if rule_cand is not None:
                    chosen = rule_cand
            elif get_action_priority(rules_action) > get_action_priority(chosen.action):
                rule_cand = next((c for c in candidates if c.action == rules_action), None)
                if rule_cand is not None:
                    chosen = rule_cand

        return NavigationDecision(
            action=chosen.action,
            reason=chosen.reason,
            confidence=confidence,
            primary_threat_id=primary_threat,
            recommended_heading=chosen.recommended_heading,
            risk_level=risk_level,
            timestamp=t,
        )
