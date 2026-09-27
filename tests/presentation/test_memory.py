"""Unit tests for presentation memory, cooldowns, and safety bypasses."""

import unittest

from app.core.enums import RiskLevel
from app.decision.model import NavigationAction
from app.presentation.memory import PresentationMemory
from app.presentation.model import (
    PresentationChannel,
    PresentationConfig,
    PresentationInstruction,
    PresentationPriority,
)


class TestPresentationMemory(unittest.TestCase):
    """Tests evaluating temporal deduplication and escalation logic."""

    def setUp(self) -> None:
        self.config = PresentationConfig(speech_cooldown=1.0, haptic_cooldown=0.5)
        self.memory = PresentationMemory(self.config)

    def test_repeated_message_suppression(self) -> None:
        inst1 = PresentationInstruction(
            channel=PresentationChannel.SPEECH,
            priority=PresentationPriority.WARNING,
            message_id="turn_left:high",
            text="Turn left",
            direction=-0.4,
            intensity=0.7,
            duration=0.8,
            timestamp=1.0,
            source_action=NavigationAction.TURN_LEFT,
            source_risk_level=RiskLevel.HIGH,
        )
        allowed, _ = self.memory.should_emit(inst1)
        self.assertTrue(allowed)
        self.memory.record_emission(inst1)

        # Immediate repeat at t=1.1s (0.1s later < 1.0s cooldown)
        inst2 = PresentationInstruction(
            channel=PresentationChannel.SPEECH,
            priority=PresentationPriority.WARNING,
            message_id="turn_left:high",
            text="Turn left",
            direction=-0.4,
            intensity=0.7,
            duration=0.8,
            timestamp=1.1,
            source_action=NavigationAction.TURN_LEFT,
            source_risk_level=RiskLevel.HIGH,
        )
        allowed2, reason2 = self.memory.should_emit(inst2)
        self.assertFalse(allowed2)
        self.assertEqual(reason2, "speech_cooldown")

    def test_emergency_bypass_cooldown(self) -> None:
        # Initial turn left at t=1.0s
        inst1 = PresentationInstruction(
            channel=PresentationChannel.SPEECH,
            priority=PresentationPriority.WARNING,
            message_id="turn_left:high",
            text="Turn left",
            direction=-0.4,
            intensity=0.7,
            duration=0.8,
            timestamp=1.0,
            source_action=NavigationAction.TURN_LEFT,
            source_risk_level=RiskLevel.HIGH,
        )
        self.memory.should_emit(inst1)
        self.memory.record_emission(inst1)

        # Emergency stop at t=1.1s (0.1s later) -> MUST BYPASS
        inst_emerg = PresentationInstruction(
            channel=PresentationChannel.SPEECH,
            priority=PresentationPriority.EMERGENCY,
            message_id="emergency_stop:critical",
            text="Stop",
            direction=None,
            intensity=1.0,
            duration=0.8,
            timestamp=1.1,
            source_action=NavigationAction.EMERGENCY_STOP,
            source_risk_level=RiskLevel.CRITICAL,
        )
        allowed, reason = self.memory.should_emit(inst_emerg)
        self.assertTrue(allowed)
        self.assertEqual(reason, "emergency_transition")


if __name__ == "__main__":
    unittest.main()
