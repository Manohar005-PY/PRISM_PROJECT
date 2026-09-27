"""Presentation engine orchestrating policy evaluation, memory filtering, and channel dispatch."""

from __future__ import annotations

from typing import Sequence

from app.core.types import RiskAssessment
from app.decision.model import NavigationDecision
from app.presentation.channels.base import PresentationChannelAdapter
from app.presentation.dispatcher import PresentationDispatcher
from app.presentation.memory import PresentationMemory
from app.presentation.model import PresentationConfig, PresentationInstruction
from app.presentation.policy import PresentationPolicy


class PresentationEngine:
    """Orchestrates end-to-end multimodal presentation from NavigationDecision to channel delivery."""

    def __init__(
        self,
        config: PresentationConfig | None = None,
        policy: PresentationPolicy | None = None,
        memory: PresentationMemory | None = None,
        dispatcher: PresentationDispatcher | None = None,
        channels: Sequence[PresentationChannelAdapter] | None = None,
    ) -> None:
        self.config = config if config is not None else PresentationConfig()
        self.policy = policy if policy is not None else PresentationPolicy()
        self.memory = memory if memory is not None else PresentationMemory(self.config)
        self.dispatcher = dispatcher if dispatcher is not None else PresentationDispatcher(channels)

        self.total_generated_count: int = 0
        self.total_emitted_count: int = 0
        self.total_suppressed_count: int = 0

    def register_channel(self, channel: PresentationChannelAdapter) -> None:
        """Register a delivery channel adapter."""
        self.dispatcher.register_channel(channel)

    def process(
        self,
        decision: NavigationDecision,
        risk_assessment: RiskAssessment | None = None,
    ) -> list[PresentationInstruction]:
        """Process an upstream decision and deliver filtered multimodal instructions.

        Args:
            decision: Authoritative NavigationDecision from DecisionEngine.
            risk_assessment: Optional evaluated RiskAssessment from Part 6.

        Returns:
            List of successfully emitted PresentationInstruction objects.
        """
        # Step 1: Generate candidate instructions for all channels
        candidates = self.policy.evaluate(decision, risk_assessment)
        self.total_generated_count += len(candidates)

        # Step 2: Temporal suppression & deduplication filtering
        self.memory.begin_frame(decision.action, decision.risk_level, decision.timestamp)
        accepted: list[PresentationInstruction] = []
        for inst in candidates:
            allowed, reason = self.memory.should_emit(inst)
            if allowed:
                self.memory.record_emission(inst)
                accepted.append(inst)
                self.total_emitted_count += 1
            else:
                self.total_suppressed_count += 1

        # Step 3: Dispatch to registered physical/mock channels
        if accepted:
            self.dispatcher.dispatch(accepted)

        return accepted

    def reset(self) -> None:
        """Reset temporal state memory."""
        self.memory.reset()
