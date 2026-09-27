"""Unit tests for presentation domain models and instructions."""

import unittest

from app.core.enums import RiskLevel
from app.decision.model import NavigationAction
from app.presentation.model import (
    HapticPattern,
    PresentationChannel,
    PresentationConfig,
    PresentationInstruction,
    PresentationPriority,
    get_priority_level,
)


class TestPresentationModel(unittest.TestCase):
    """Tests evaluating presentation models, validation, and immutability."""

    def test_instruction_creation_and_immutability(self) -> None:
        inst = PresentationInstruction(
            channel=PresentationChannel.SPEECH,
            priority=PresentationPriority.URGENT,
            message_id="turn_left:high",
            text="Turn left",
            direction=-0.45,
            intensity=0.8,
            duration=0.8,
            timestamp=1.0,
            source_action=NavigationAction.TURN_LEFT,
            source_risk_level=RiskLevel.HIGH,
            threat_id=1,
            haptic_pattern=None,
        )
        self.assertEqual(inst.channel, PresentationChannel.SPEECH)
        self.assertEqual(inst.priority, PresentationPriority.URGENT)
        self.assertEqual(inst.text, "Turn left")
        self.assertEqual(inst.direction, -0.45)
        self.assertEqual(inst.intensity, 0.8)

        # Immutability
        with self.assertRaises(Exception):
            inst.text = "Other"  # type: ignore[misc]

    def test_invalid_parameters_raise_validation_errors(self) -> None:
        # Invalid channel
        with self.assertRaises(TypeError):
            PresentationInstruction(
                channel="audio",  # type: ignore[arg-type]
                priority=PresentationPriority.INFO,
                message_id="m1",
                text=None,
                direction=None,
                intensity=0.5,
                duration=1.0,
                timestamp=0.0,
                source_action=NavigationAction.CONTINUE,
                source_risk_level=RiskLevel.SAFE,
            )

        # Invalid intensity (>1.0)
        with self.assertRaises(ValueError):
            PresentationInstruction(
                channel=PresentationChannel.SPEECH,
                priority=PresentationPriority.INFO,
                message_id="m1",
                text="test",
                direction=None,
                intensity=1.5,
                duration=1.0,
                timestamp=0.0,
                source_action=NavigationAction.CONTINUE,
                source_risk_level=RiskLevel.SAFE,
            )

    def test_priority_ranking(self) -> None:
        self.assertLess(get_priority_level(PresentationPriority.INFO), get_priority_level(PresentationPriority.CAUTION))
        self.assertLess(get_priority_level(PresentationPriority.WARNING), get_priority_level(PresentationPriority.URGENT))
        self.assertLess(get_priority_level(PresentationPriority.URGENT), get_priority_level(PresentationPriority.EMERGENCY))


if __name__ == "__main__":
    unittest.main()
