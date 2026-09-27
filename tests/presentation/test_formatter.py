"""Unit tests for speech and message formatting."""

import unittest

from app.core.enums import RiskLevel
from app.decision.model import NavigationAction
from app.presentation.formatter import SpeechFormatter


class TestSpeechFormatter(unittest.TestCase):
    """Tests evaluating concise speech and message ID generation."""

    def test_concise_speech_generation(self) -> None:
        self.assertEqual(SpeechFormatter.format_speech(NavigationAction.STOP, RiskLevel.HIGH), "Stop")
        self.assertEqual(SpeechFormatter.format_speech(NavigationAction.EMERGENCY_STOP, RiskLevel.CRITICAL), "Stop")
        self.assertEqual(SpeechFormatter.format_speech(NavigationAction.SLOW_DOWN, RiskLevel.MEDIUM), "Slow down")
        self.assertEqual(SpeechFormatter.format_speech(NavigationAction.TURN_LEFT, RiskLevel.MEDIUM), "Turn left")
        self.assertEqual(SpeechFormatter.format_speech(NavigationAction.TURN_RIGHT, RiskLevel.MEDIUM), "Turn right")
        self.assertEqual(SpeechFormatter.format_speech(NavigationAction.CAUTION, RiskLevel.LOW), "Caution ahead")
        self.assertEqual(SpeechFormatter.format_speech(NavigationAction.CONTINUE, RiskLevel.SAFE), "")


if __name__ == "__main__":
    unittest.main()
