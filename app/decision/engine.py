"""Main DecisionEngine orchestrator and Part 1 protocol adapter."""

from __future__ import annotations

from app.core.enums import DecisionType, RiskLevel
from app.core.interfaces import DecisionEngine as DecisionEngineProtocol
from app.core.types import Decision, RiskAssessment
from app.decision.arbitrator import DecisionArbitrator, DecisionMemory
from app.decision.context import DecisionContext
from app.decision.heading import compute_safe_headings, select_best_heading
from app.decision.model import (
    DecisionReason,
    NavigationAction,
    NavigationDecision,
)
from app.decision.planner import LocalMotionPlanner
from app.decision.rules import evaluate_safety_rules


class DecisionEngine:
    """Orchestrates local avoidance planning, safety rules, clearance analysis, and arbitration.

    Conforms to the app.core.interfaces.DecisionEngine Protocol via an adapter interface.
    """

    def __init__(
        self,
        planner: LocalMotionPlanner | None = None,
        arbitrator: DecisionArbitrator | None = None,
        memory: DecisionMemory | None = None,
    ) -> None:
        self.planner: LocalMotionPlanner = planner if planner is not None else LocalMotionPlanner()
        self.arbitrator: DecisionArbitrator = arbitrator if arbitrator is not None else DecisionArbitrator()
        self.memory: DecisionMemory = memory if memory is not None else DecisionMemory()

    def decide_navigation(self, context: DecisionContext) -> NavigationDecision:
        """Evaluate full decision pipeline from DecisionContext to authoritative NavigationDecision.

        Pipeline:
            DecisionContext
                ↓
            Safe Heading Clearance Ray Casting
                ↓
            Best Heading Selection
                ↓
            Deterministic Safety Rules
                ↓
            Candidate Action Generation & Hard Safety Filtering
                ↓
            Cost Evaluation & Ranking
                ↓
            Arbitration & Lateral Hysteresis
                ↓
            NavigationDecision

        Args:
            context: Comprehensive DecisionContext containing spatial state and risk inputs.

        Returns:
            Authoritative NavigationDecision instance.
        """
        # Step 1: Geometric clearance analysis across candidate angular directions
        safe_headings = compute_safe_headings(context)
        best_heading = select_best_heading(safe_headings, context)

        # Step 2: Evaluate deterministic safety rules
        risk_level = context.risk_assessment.risk_level if context.risk_assessment is not None else RiskLevel.SAFE
        rules_action, rules_reason = evaluate_safety_rules(risk_level, best_heading, context)

        # Step 3: Local motion planning with hard constraints and soft costs
        candidates = self.planner.plan_candidates(context, safe_headings)

        # Step 4: Arbitration with safety escalation and lateral hysteresis
        decision = self.arbitrator.arbitrate(
            candidates=candidates,
            rules_action=rules_action,
            rules_reason=rules_reason,
            context=context,
            memory=self.memory,
        )

        # Step 5: Update state memory
        self.memory.update(decision)

        return decision

    def decide(self, risk_assessment: RiskAssessment) -> Decision:
        """Part 1 Protocol Adapter: Convert RiskAssessment to core Decision struct.

        Implements app.core.interfaces.DecisionEngine.
        """
        context = DecisionContext(
            risk_assessment=risk_assessment,
            timestamp=risk_assessment.timestamp,
        )
        nav_decision = self.decide_navigation(context)

        # Map NavigationAction to Part 1 DecisionType
        if nav_decision.action in (NavigationAction.EMERGENCY_STOP, NavigationAction.STOP):
            dt = DecisionType.STOP
        elif nav_decision.action in (NavigationAction.TURN_LEFT, NavigationAction.TURN_RIGHT, NavigationAction.SLOW_DOWN):
            dt = DecisionType.WARNING
        elif nav_decision.action == NavigationAction.CAUTION:
            dt = DecisionType.CAUTION
        else:
            dt = DecisionType.NO_ACTION

        return Decision(
            decision_type=dt,
            object_id=risk_assessment.object_id,
            timestamp=risk_assessment.timestamp,
            message_id=nav_decision.reason.value,
        )
