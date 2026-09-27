"""Deterministic natural language and message formatting for the presentation subsystem."""

from __future__ import annotations

from app.core.enums import RiskLevel
from app.decision.model import DecisionReason, NavigationAction


class SpeechFormatter:
    """Centralizes concise, deterministic natural-language utterances for assistive navigation."""

    @staticmethod
    def format_speech(action: NavigationAction, risk_level: RiskLevel) -> str:
        """Generate concise speech utterance corresponding to a navigation decision."""
        if action == NavigationAction.EMERGENCY_STOP:
            return "Stop"
        elif action == NavigationAction.STOP:
            return "Stop"
        elif action == NavigationAction.SLOW_DOWN:
            return "Slow down"
        elif action == NavigationAction.TURN_LEFT:
            return "Turn left"
        elif action == NavigationAction.TURN_RIGHT:
            return "Turn right"
        elif action == NavigationAction.CAUTION:
            return "Caution ahead"
        elif action == NavigationAction.CONTINUE:
            return ""
        return ""

    @staticmethod
    def generate_message_id(action: NavigationAction, risk_level: RiskLevel) -> str:
        """Produce deterministic semantic key for deduplication and cooldown tracking."""
        return f"{action.value}:{risk_level.value}"

    @staticmethod
    def format_visual_banner(action: NavigationAction, reason: DecisionReason, risk_level: RiskLevel) -> str:
        """Produce visual HUD banner text."""
        return f"[{risk_level.value.upper()}] {action.value.upper()} ({reason.value})"
