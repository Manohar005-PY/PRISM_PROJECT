"""Integration tests executing complete Decision -> Presentation -> Channel delivery pipeline."""

import unittest

from app.core.enums import RiskLevel
from app.core.types import RiskAssessment
from app.decision.engine import DecisionEngine
from app.decision.model import DecisionReason, NavigationAction, NavigationDecision
from app.presentation.channels.mock import (
    MockHapticChannel,
    MockSpatialAudioChannel,
    MockSpeechChannel,
    MockVisualChannel,
)
from app.presentation.engine import PresentationEngine
from app.presentation.model import (
    HapticPattern,
    PresentationChannel,
    PresentationConfig,
)


class TestPresentationIntegration(unittest.TestCase):
    """End-to-end integration test of the presentation subsystem."""

    def setUp(self) -> None:
        self.speech_ch = MockSpeechChannel()
        self.audio_ch = MockSpatialAudioChannel()
        self.haptic_ch = MockHapticChannel()
        self.visual_ch = MockVisualChannel()

        self.config = PresentationConfig(speech_cooldown=1.5, haptic_cooldown=0.8)
        self.engine = PresentationEngine(
            config=self.config,
            channels=[self.speech_ch, self.audio_ch, self.haptic_ch, self.visual_ch],
        )

    def test_full_scenario_escalation_flow(self) -> None:
        """Trace full sequential transition from Safe -> Turn Left -> Emergency Stop."""
        # 1. Frame 1: SAFE
        dec1 = NavigationDecision(
            action=NavigationAction.CONTINUE,
            reason=DecisionReason.NO_HAZARD,
            confidence=1.0,
            primary_threat_id=None,
            recommended_heading=0.0,
            risk_level=RiskLevel.SAFE,
            timestamp=0.0,
        )
        cues1 = self.engine.process(dec1)
        self.assertEqual(len(cues1), 1)  # Only visual
        self.assertEqual(len(self.speech_ch.delivered_instructions), 0)

        # 2. Frame 2: TURN_LEFT at t=1.0s
        dec2 = NavigationDecision(
            action=NavigationAction.TURN_LEFT,
            reason=DecisionReason.SAFE_PATH_AVAILABLE,
            confidence=0.95,
            primary_threat_id=1,
            recommended_heading=-0.42,
            risk_level=RiskLevel.HIGH,
            timestamp=1.0,
        )
        cues2 = self.engine.process(dec2)
        self.assertEqual(len(cues2), 4)  # Speech, Audio, Haptic, Visual
        self.assertEqual(len(self.speech_ch.delivered_instructions), 1)
        self.assertEqual(self.speech_ch.delivered_instructions[0].text, "Turn left")

        # 3. Frame 3: REPEATED TURN_LEFT at t=1.1s (0.1s later)
        dec3 = NavigationDecision(
            action=NavigationAction.TURN_LEFT,
            reason=DecisionReason.SAFE_PATH_AVAILABLE,
            confidence=0.95,
            primary_threat_id=1,
            recommended_heading=-0.42,
            risk_level=RiskLevel.HIGH,
            timestamp=1.1,
        )
        cues3 = self.engine.process(dec3)
        self.assertEqual(len(cues3), 1)  # Only visual HUD updates, speech/audio/haptic suppressed!
        self.assertEqual(len(self.speech_ch.delivered_instructions), 1)

        # 4. Frame 4: EMERGENCY_STOP at t=1.2s (0.1s later -> MUST BYPASS COOLDOWN!)
        dec4 = NavigationDecision(
            action=NavigationAction.EMERGENCY_STOP,
            reason=DecisionReason.EMERGENCY_COLLISION_RISK,
            confidence=0.99,
            primary_threat_id=1,
            recommended_heading=None,
            risk_level=RiskLevel.CRITICAL,
            timestamp=1.2,
        )
        cues4 = self.engine.process(dec4)
        self.assertEqual(len(cues4), 4)  # Speech, Audio, Haptic, Visual
        self.assertEqual(len(self.speech_ch.delivered_instructions), 2)
        self.assertEqual(self.speech_ch.delivered_instructions[1].text, "Stop")
        self.assertEqual(len(self.audio_ch.delivered_instructions), 2)
        self.assertEqual(self.audio_ch.delivered_instructions[1].direction, 0.0)
        self.assertEqual(self.haptic_ch.delivered_instructions[-1].haptic_pattern, HapticPattern.EMERGENCY)


if __name__ == "__main__":
    unittest.main()
