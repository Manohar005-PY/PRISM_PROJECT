"""Presentation memory, temporal deduplication, cooldowns, and safety escalation bypasses."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.enums import RiskLevel
from app.decision.model import NavigationAction
from app.presentation.model import (
    PresentationChannel,
    PresentationConfig,
    PresentationInstruction,
    PresentationPriority,
)


def get_risk_rank(risk: RiskLevel) -> int:
    """Integer rank for risk escalation comparison."""
    ranks = {
        RiskLevel.SAFE: 1,
        RiskLevel.LOW: 2,
        RiskLevel.MEDIUM: 3,
        RiskLevel.HIGH: 4,
        RiskLevel.CRITICAL: 5,
    }
    return ranks.get(risk, 1)


class PresentationMemory:
    """Tracks historical presentation state and arbitrates temporal suppression across multimodal channels."""

    def __init__(self, config: PresentationConfig | None = None) -> None:
        self.config = config if config is not None else PresentationConfig()
        # Key: channel -> last_emission_timestamp
        self._last_channel_emission: dict[PresentationChannel, float] = {}
        # Historical frame states
        self._last_emitted_action: NavigationAction | None = None
        self._last_emitted_risk_level: RiskLevel | None = None
        self._current_frame_timestamp: float | None = None
        self._frame_is_action_transition: bool = False
        self._frame_is_risk_escalation: bool = False
        self._frame_is_emergency_transition: bool = False

    def reset(self) -> None:
        """Reset memory state."""
        self._last_channel_emission.clear()
        self._last_emitted_action = None
        self._last_emitted_risk_level = None
        self._current_frame_timestamp = None
        self._frame_is_action_transition = False
        self._frame_is_risk_escalation = False
        self._frame_is_emergency_transition = False

    def begin_frame(self, action: NavigationAction, risk_level: RiskLevel, timestamp: float) -> None:
        """Initialize frame-level transition flags before evaluating individual channel instructions."""
        self._current_frame_timestamp = timestamp
        self._frame_is_action_transition = (
            self._last_emitted_action is not None and action != self._last_emitted_action
        )
        self._frame_is_risk_escalation = (
            self.config.risk_escalation_bypass
            and self._last_emitted_risk_level is not None
            and get_risk_rank(risk_level) > get_risk_rank(self._last_emitted_risk_level)
        )
        self._frame_is_emergency_transition = (
            self.config.emergency_bypass
            and (action == NavigationAction.EMERGENCY_STOP or risk_level == RiskLevel.CRITICAL)
            and (self._last_emitted_action != NavigationAction.EMERGENCY_STOP)
        )

    def should_emit(
        self,
        instruction: PresentationInstruction,
    ) -> tuple[bool, str]:
        """Determine whether an instruction should be emitted or suppressed by temporal policy.

        Returns:
            (allow_emission: bool, reason: str)
        """
        now = instruction.timestamp
        channel = instruction.channel
        action = instruction.source_action
        risk = instruction.source_risk_level
        prio = instruction.priority

        # Auto-sync frame if begin_frame was not explicitly called
        if self._current_frame_timestamp != now:
            self.begin_frame(action, risk, now)

        # --------------------------------------------------------------------
        # 1. EMERGENCY TRANSITION & PERIODIC REPETITION
        # --------------------------------------------------------------------
        if self.config.emergency_bypass and (
            prio == PresentationPriority.EMERGENCY or action == NavigationAction.EMERGENCY_STOP
        ):
            if self._frame_is_emergency_transition:
                return True, "emergency_transition"

            last_ch_t = self._last_channel_emission.get(channel, -1.0)
            if last_ch_t < 0.0 or (now - last_ch_t) >= 0.6:
                return True, "emergency_periodic"
            return False, "emergency_cooldown"

        # --------------------------------------------------------------------
        # 2. ACTION TRANSITION BYPASS
        # --------------------------------------------------------------------
        if self._frame_is_action_transition:
            return True, "action_transition"

        # --------------------------------------------------------------------
        # 3. RISK ESCALATION BYPASS
        # --------------------------------------------------------------------
        if self._frame_is_risk_escalation:
            return True, "risk_escalation"

        # --------------------------------------------------------------------
        # 4. CHANNEL-SPECIFIC COOLDOWN
        # --------------------------------------------------------------------
        last_t = self._last_channel_emission.get(channel, -1.0)
        cooldown = self._get_channel_cooldown(channel)

        if last_t >= 0.0 and (now - last_t) < cooldown:
            return False, f"{channel.value}_cooldown"

        return True, "allowed"

    def record_emission(self, instruction: PresentationInstruction) -> None:
        """Update memory with accepted instruction."""
        now = instruction.timestamp
        self._last_channel_emission[instruction.channel] = now
        self._last_emitted_action = instruction.source_action
        self._last_emitted_risk_level = instruction.source_risk_level

    def _get_channel_cooldown(self, channel: PresentationChannel) -> float:
        if channel == PresentationChannel.SPEECH:
            return self.config.speech_cooldown
        elif channel == PresentationChannel.SPATIAL_AUDIO:
            return self.config.spatial_audio_cooldown
        elif channel == PresentationChannel.HAPTIC:
            return self.config.haptic_cooldown
        elif channel == PresentationChannel.VISUAL:
            return 0.0  # Visual HUD updates every frame
        return 1.0
