"""Unit tests for presentation dispatcher and channel failure isolation."""

import unittest

from app.core.enums import RiskLevel
from app.decision.model import NavigationAction
from app.presentation.channels.mock import MockSpeechChannel, MockVisualChannel
from app.presentation.dispatcher import PresentationDispatcher
from app.presentation.model import (
    PresentationChannel,
    PresentationInstruction,
    PresentationPriority,
)


class TestPresentationDispatcher(unittest.TestCase):
    """Tests evaluating routing, delivery, and fault isolation."""

    def test_routing_to_registered_channels(self) -> None:
        speech = MockSpeechChannel()
        visual = MockVisualChannel()
        dispatcher = PresentationDispatcher([speech, visual])

        inst1 = PresentationInstruction(
            channel=PresentationChannel.SPEECH,
            priority=PresentationPriority.WARNING,
            message_id="m1",
            text="Slow down",
            direction=None,
            intensity=0.7,
            duration=0.8,
            timestamp=1.0,
            source_action=NavigationAction.SLOW_DOWN,
            source_risk_level=RiskLevel.MEDIUM,
        )
        inst2 = PresentationInstruction(
            channel=PresentationChannel.VISUAL,
            priority=PresentationPriority.WARNING,
            message_id="m1",
            text="SLOW DOWN",
            direction=None,
            intensity=0.7,
            duration=0.5,
            timestamp=1.0,
            source_action=NavigationAction.SLOW_DOWN,
            source_risk_level=RiskLevel.MEDIUM,
        )

        res = dispatcher.dispatch([inst1, inst2])
        self.assertTrue(res[PresentationChannel.SPEECH])
        self.assertTrue(res[PresentationChannel.VISUAL])
        self.assertEqual(len(speech.delivered_instructions), 1)
        self.assertEqual(len(visual.delivered_instructions), 1)

    def test_channel_failure_does_not_crash_other_channels(self) -> None:
        speech = MockSpeechChannel(available=False)  # Broken channel
        visual = MockVisualChannel(available=True)
        dispatcher = PresentationDispatcher([speech, visual])

        inst1 = PresentationInstruction(
            channel=PresentationChannel.SPEECH,
            priority=PresentationPriority.WARNING,
            message_id="m1",
            text="Stop",
            direction=None,
            intensity=0.9,
            duration=0.8,
            timestamp=1.0,
            source_action=NavigationAction.STOP,
            source_risk_level=RiskLevel.HIGH,
        )
        inst2 = PresentationInstruction(
            channel=PresentationChannel.VISUAL,
            priority=PresentationPriority.WARNING,
            message_id="m1",
            text="STOP",
            direction=None,
            intensity=0.9,
            duration=0.5,
            timestamp=1.0,
            source_action=NavigationAction.STOP,
            source_risk_level=RiskLevel.HIGH,
        )

        res = dispatcher.dispatch([inst1, inst2])
        self.assertFalse(res[PresentationChannel.SPEECH])
        self.assertTrue(res[PresentationChannel.VISUAL])
        self.assertEqual(dispatcher.failure_count, 1)


if __name__ == "__main__":
    unittest.main()
