"""Unit tests for presentation policy mapping."""

import unittest

from app.core.enums import RiskLevel
from app.decision.model import DecisionReason, NavigationAction, NavigationDecision
from app.presentation.model import HapticPattern, PresentationChannel, PresentationPriority
from app.presentation.policy import PresentationPolicy


class TestPresentationPolicy(unittest.TestCase):
    """Tests evaluating deterministic policy mapping from decisions to cues."""

    def setUp(self) -> None:
        self.policy = PresentationPolicy()

    def test_continue_mapping(self) -> None:
        dec = NavigationDecision(
            action=NavigationAction.CONTINUE,
            reason=DecisionReason.NO_HAZARD,
            confidence=1.0,
            primary_threat_id=None,
            recommended_heading=0.0,
            risk_level=RiskLevel.SAFE,
            timestamp=0.0,
        )
        cues = self.policy.evaluate(dec)
        channels = [c.channel for c in cues]
        self.assertIn(PresentationChannel.VISUAL, channels)
        self.assertNotIn(PresentationChannel.SPEECH, channels)
        self.assertNotIn(PresentationChannel.HAPTIC, channels)

    def test_emergency_stop_mapping(self) -> None:
        dec = NavigationDecision(
            action=NavigationAction.EMERGENCY_STOP,
            reason=DecisionReason.EMERGENCY_COLLISION_RISK,
            confidence=0.99,
            primary_threat_id=1,
            recommended_heading=None,
            risk_level=RiskLevel.CRITICAL,
            timestamp=1.0,
        )
        cues = self.policy.evaluate(dec)
        channels = {c.channel: c for c in cues}
        self.assertIn(PresentationChannel.SPEECH, channels)
        self.assertIn(PresentationChannel.SPATIAL_AUDIO, channels)
        self.assertIn(PresentationChannel.HAPTIC, channels)
        self.assertEqual(channels[PresentationChannel.SPEECH].text, "Stop")
        self.assertEqual(channels[PresentationChannel.SPEECH].priority, PresentationPriority.EMERGENCY)
        self.assertEqual(channels[PresentationChannel.SPATIAL_AUDIO].direction, 0.0)
        self.assertEqual(channels[PresentationChannel.HAPTIC].haptic_pattern, HapticPattern.EMERGENCY)

    def test_turn_left_mapping(self) -> None:
        dec = NavigationDecision(
            action=NavigationAction.TURN_LEFT,
            reason=DecisionReason.SAFE_PATH_AVAILABLE,
            confidence=0.95,
            primary_threat_id=1,
            recommended_heading=-0.42,
            risk_level=RiskLevel.HIGH,
            timestamp=2.0,
        )
        cues = self.policy.evaluate(dec)
        channels = {c.channel: c for c in cues}
        self.assertIn(PresentationChannel.SPEECH, channels)
        self.assertIn(PresentationChannel.SPATIAL_AUDIO, channels)
        self.assertIn(PresentationChannel.HAPTIC, channels)
        self.assertEqual(channels[PresentationChannel.SPEECH].text, "Turn left")
        self.assertAlmostEqual(channels[PresentationChannel.SPATIAL_AUDIO].direction or 0.0, -0.42, places=2)
        self.assertEqual(channels[PresentationChannel.HAPTIC].haptic_pattern, HapticPattern.LEFT)


if __name__ == "__main__":
    unittest.main()
