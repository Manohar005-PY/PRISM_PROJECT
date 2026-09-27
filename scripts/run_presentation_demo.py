"""PRISM Multimodal Presentation Engine Demonstration Script.

Demonstrates deterministic mapping of NavigationDecisions and RiskAssessments into multimodal cues
(Speech, Spatial Audio, Haptic, Visual), verifying temporal cooldowns, emergency escalation bypasses,
and channel failure isolation.
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.enums import RiskLevel
from app.core.types import RiskAssessment
from app.decision.model import DecisionReason, NavigationAction, NavigationDecision
from app.presentation.channels.mock import (
    MockHapticChannel,
    MockSpatialAudioChannel,
    MockSpeechChannel,
    MockVisualChannel,
)
from app.presentation.engine import PresentationEngine
from app.presentation.model import (
    PresentationChannel,
    PresentationConfig,
    PresentationInstruction,
)


def format_instruction_summary(inst: PresentationInstruction) -> str:
    """Format a single instruction for readable CLI output."""
    ch = inst.channel.value.upper()
    if inst.channel == PresentationChannel.SPEECH:
        return f"{ch:<14} -> \"{inst.text}\" (prio={inst.priority.value}, int={inst.intensity:.2f})"
    elif inst.channel == PresentationChannel.SPATIAL_AUDIO:
        ang_deg = (inst.direction * 180.0 / 3.14159) if inst.direction is not None else 0.0
        return f"{ch:<14} -> angle={inst.direction:+.2f} rad ({ang_deg:+.1f} deg), int={inst.intensity:.2f}"
    elif inst.channel == PresentationChannel.HAPTIC:
        pat = inst.haptic_pattern.value if inst.haptic_pattern else "NONE"
        return f"{ch:<14} -> pattern={pat.upper()}, int={inst.intensity:.2f}, dur={inst.duration:.1f}s"
    elif inst.channel == PresentationChannel.VISUAL:
        return f"{ch:<14} -> banner=\"{inst.text}\""
    return f"{ch:<14} -> {inst.message_id}"


def main() -> None:
    print("=" * 80)
    print(" PRISM MULTIMODAL PRESENTATION ENGINE DEMONSTRATION")
    print("=" * 80 + "\n")

    # 1. Initialize Mock Channels & Engine
    speech_ch = MockSpeechChannel()
    audio_ch = MockSpatialAudioChannel()
    haptic_ch = MockHapticChannel()
    visual_ch = MockVisualChannel()

    engine = PresentationEngine(
        config=PresentationConfig(speech_cooldown=1.5, haptic_cooldown=0.8, repeated_action_cooldown=2.0),
        channels=[speech_ch, audio_ch, haptic_ch, visual_ch],
    )

    scenarios = [
        (
            "1. SAFE CRUISE",
            NavigationDecision(
                action=NavigationAction.CONTINUE,
                reason=DecisionReason.NO_HAZARD,
                confidence=1.0,
                primary_threat_id=None,
                recommended_heading=0.0,
                risk_level=RiskLevel.SAFE,
                timestamp=0.0,
            ),
            None,
        ),
        (
            "2. CAUTION ADVISORY",
            NavigationDecision(
                action=NavigationAction.CAUTION,
                reason=DecisionReason.LOW_COLLISION_RISK,
                confidence=0.9,
                primary_threat_id=1,
                recommended_heading=0.0,
                risk_level=RiskLevel.LOW,
                timestamp=0.5,
            ),
            RiskAssessment(1, RiskLevel.LOW, 25.0, 15.0, 1.2, 0.9, 0.5),
        ),
        (
            "3. SLOW DOWN ADVISORY",
            NavigationDecision(
                action=NavigationAction.SLOW_DOWN,
                reason=DecisionReason.MEDIUM_COLLISION_RISK,
                confidence=0.92,
                primary_threat_id=1,
                recommended_heading=0.0,
                risk_level=RiskLevel.MEDIUM,
                timestamp=1.0,
            ),
            RiskAssessment(1, RiskLevel.MEDIUM, 50.0, 4.5, 0.6, 0.92, 1.0),
        ),
        (
            "4. LATERAL EVASION (TURN LEFT)",
            NavigationDecision(
                action=NavigationAction.TURN_LEFT,
                reason=DecisionReason.SAFE_PATH_AVAILABLE,
                confidence=0.95,
                primary_threat_id=1,
                recommended_heading=-0.42,
                risk_level=RiskLevel.HIGH,
                timestamp=1.5,
            ),
            RiskAssessment(1, RiskLevel.HIGH, 75.0, 2.5, 0.2, 0.95, 1.5),
        ),
        (
            "5. REPEATED TURN LEFT (TEMPORAL SUPPRESSION CHECK)",
            NavigationDecision(
                action=NavigationAction.TURN_LEFT,
                reason=DecisionReason.SAFE_PATH_AVAILABLE,
                confidence=0.95,
                primary_threat_id=1,
                recommended_heading=-0.42,
                risk_level=RiskLevel.HIGH,
                timestamp=1.6,  # 0.1s later -> should be suppressed
            ),
            RiskAssessment(1, RiskLevel.HIGH, 76.0, 2.4, 0.2, 0.95, 1.6),
        ),
        (
            "6. EMERGENCY STOP (SAFETY ESCALATION BYPASS)",
            NavigationDecision(
                action=NavigationAction.EMERGENCY_STOP,
                reason=DecisionReason.EMERGENCY_COLLISION_RISK,
                confidence=0.99,
                primary_threat_id=1,
                recommended_heading=None,
                risk_level=RiskLevel.CRITICAL,
                timestamp=1.7,  # 0.1s later -> MUST BYPASS COOLDOWN
            ),
            RiskAssessment(1, RiskLevel.CRITICAL, 98.0, 0.5, 0.0, 0.99, 1.7),
        ),
    ]

    for title, dec, risk in scenarios:
        print("-" * 80)
        print(f"Scenario: {title}")
        print(f"Decision:  Action={dec.action.value.upper()}, Risk={dec.risk_level.value.upper()}, Reason={dec.reason.value}")
        emitted = engine.process(dec, risk)
        if emitted:
            print("Emitted Multimodal Cues:")
            for inst in emitted:
                print(f"  * {format_instruction_summary(inst)}")
        else:
            print("Emitted Multimodal Cues: [SUPPRESSED BY COOLDOWN POLICY]")
        print()

    # 7. Test Channel Failure Isolation
    print("-" * 80)
    print("Scenario: 7. CHANNEL FAILURE ISOLATION (Broken Speech Driver)")
    speech_ch.set_available(False)  # Simulate speech audio hardware error
    dec_test = NavigationDecision(
        action=NavigationAction.TURN_RIGHT,
        reason=DecisionReason.SAFE_PATH_AVAILABLE,
        confidence=0.95,
        primary_threat_id=2,
        recommended_heading=0.45,
        risk_level=RiskLevel.HIGH,
        timestamp=4.0,
    )
    emitted = engine.process(dec_test, None)
    print("Decision: Action=TURN_RIGHT, Risk=HIGH")
    print("Emitted Multimodal Cues (Speech channel failed, other channels succeeded):")
    for inst in emitted:
        print(f"  * {format_instruction_summary(inst)}")

    print("\n" + "=" * 80)
    print("DEMONSTRATION SUMMARY:")
    print(f" - Total Generated Instructions: {engine.total_generated_count}")
    print(f" - Total Emitted Instructions:  {engine.total_emitted_count}")
    print(f" - Total Suppressed:            {engine.total_suppressed_count}")
    print(f" - Channel Failures Handled:    {engine.dispatcher.failure_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
