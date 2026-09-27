"""Unit tests for concrete presentation channel adapters."""

import unittest

from app.core.enums import RiskLevel
from app.decision.model import NavigationAction
from app.presentation.channels.haptic import HapticChannelAdapter
from app.presentation.channels.spatial_audio import SpatialAudioChannelAdapter
from app.presentation.channels.speech import TextSpeechChannel
from app.presentation.channels.visual import VisualChannelAdapter
from app.presentation.model import (
    HapticPattern,
    PresentationChannel,
    PresentationInstruction,
    PresentationPriority,
)


class TestPresentationChannels(unittest.TestCase):
    """Tests evaluating delivery by concrete channel adapters."""

    def test_text_speech_channel(self) -> None:
        speech = TextSpeechChannel()
        inst = PresentationInstruction(
            channel=PresentationChannel.SPEECH,
            priority=PresentationPriority.WARNING,
            message_id="m1",
            text="Turn left",
            direction=-0.4,
            intensity=0.7,
            duration=0.8,
            timestamp=1.0,
            source_action=NavigationAction.TURN_LEFT,
            source_risk_level=RiskLevel.HIGH,
        )
        self.assertTrue(speech.deliver(inst))
        self.assertEqual(speech.utterance_history, ["Turn left"])

    def test_spatial_audio_channel(self) -> None:
        audio = SpatialAudioChannelAdapter()
        inst = PresentationInstruction(
            channel=PresentationChannel.SPATIAL_AUDIO,
            priority=PresentationPriority.WARNING,
            message_id="m1",
            text=None,
            direction=0.45,
            intensity=0.8,
            duration=0.6,
            timestamp=1.0,
            source_action=NavigationAction.TURN_RIGHT,
            source_risk_level=RiskLevel.HIGH,
        )
        self.assertTrue(audio.deliver(inst))
        self.assertEqual(len(audio.cues), 1)
        self.assertAlmostEqual(audio.cues[0].direction_rad, 0.45, places=2)

    def test_haptic_channel(self) -> None:
        haptic = HapticChannelAdapter()
        inst = PresentationInstruction(
            channel=PresentationChannel.HAPTIC,
            priority=PresentationPriority.EMERGENCY,
            message_id="m1",
            text=None,
            direction=None,
            intensity=1.0,
            duration=1.0,
            timestamp=1.0,
            source_action=NavigationAction.EMERGENCY_STOP,
            source_risk_level=RiskLevel.CRITICAL,
            haptic_pattern=HapticPattern.EMERGENCY,
        )
        self.assertTrue(haptic.deliver(inst))
        self.assertEqual(len(haptic.actuations), 1)
        self.assertEqual(haptic.actuations[0].pattern, HapticPattern.EMERGENCY)


if __name__ == "__main__":
    unittest.main()
