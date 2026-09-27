"""Deterministic presentation policy mapping NavigationDecisions to multimodal cues."""

from __future__ import annotations

from app.core.enums import RiskLevel
from app.core.types import RiskAssessment
from app.decision.model import NavigationAction, NavigationDecision
from app.presentation.formatter import SpeechFormatter
from app.presentation.model import (
    HapticPattern,
    PresentationChannel,
    PresentationInstruction,
    PresentationPriority,
)


class PresentationPolicy:
    """Evaluates upstream decisions and risk to synthesize structured PresentationInstructions."""

    def evaluate(
        self,
        decision: NavigationDecision,
        risk_assessment: RiskAssessment | None = None,
    ) -> list[PresentationInstruction]:
        """Generate candidate presentation instructions for all active modalities.

        Args:
            decision: Authoritative NavigationDecision from DecisionEngine.
            risk_assessment: Optional evaluated RiskAssessment from Part 6.

        Returns:
            List of generated PresentationInstruction instances.
        """
        instructions: list[PresentationInstruction] = []
        action = decision.action
        risk_lvl = decision.risk_level
        threat_id = decision.primary_threat_id
        t = decision.timestamp
        heading = decision.recommended_heading
        msg_id = SpeechFormatter.generate_message_id(action, risk_lvl)

        # --------------------------------------------------------------------
        # 1. EMERGENCY_STOP / STOP
        # --------------------------------------------------------------------
        if action == NavigationAction.EMERGENCY_STOP:
            # Speech
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.SPEECH,
                    priority=PresentationPriority.EMERGENCY,
                    message_id=msg_id,
                    text="Stop",
                    direction=None,
                    intensity=1.0,
                    duration=0.8,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                )
            )
            # Spatial Audio (Forward Urgent Beacon)
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.SPATIAL_AUDIO,
                    priority=PresentationPriority.EMERGENCY,
                    message_id=msg_id,
                    text=None,
                    direction=0.0,
                    intensity=1.0,
                    duration=0.8,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                )
            )
            # Haptic
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.HAPTIC,
                    priority=PresentationPriority.EMERGENCY,
                    message_id=msg_id,
                    text=None,
                    direction=None,
                    intensity=1.0,
                    duration=1.0,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                    haptic_pattern=HapticPattern.EMERGENCY,
                )
            )
            # Visual
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.VISUAL,
                    priority=PresentationPriority.EMERGENCY,
                    message_id=msg_id,
                    text=SpeechFormatter.format_visual_banner(action, decision.reason, risk_lvl),
                    direction=None,
                    intensity=1.0,
                    duration=0.5,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                )
            )
            return instructions

        if action == NavigationAction.STOP:
            # Speech
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.SPEECH,
                    priority=PresentationPriority.URGENT,
                    message_id=msg_id,
                    text="Stop",
                    direction=None,
                    intensity=0.9,
                    duration=0.8,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                )
            )
            # Spatial Audio (Forward Urgent Beacon)
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.SPATIAL_AUDIO,
                    priority=PresentationPriority.URGENT,
                    message_id=msg_id,
                    text=None,
                    direction=0.0,
                    intensity=0.9,
                    duration=0.8,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                )
            )
            # Haptic
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.HAPTIC,
                    priority=PresentationPriority.URGENT,
                    message_id=msg_id,
                    text=None,
                    direction=None,
                    intensity=0.85,
                    duration=0.8,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                    haptic_pattern=HapticPattern.STOP,
                )
            )
            # Visual
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.VISUAL,
                    priority=PresentationPriority.URGENT,
                    message_id=msg_id,
                    text=SpeechFormatter.format_visual_banner(action, decision.reason, risk_lvl),
                    direction=None,
                    intensity=0.9,
                    duration=0.5,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                )
            )
            return instructions

        # --------------------------------------------------------------------
        # 2. SLOW_DOWN
        # --------------------------------------------------------------------
        if action == NavigationAction.SLOW_DOWN:
            prio = PresentationPriority.URGENT if risk_lvl == RiskLevel.HIGH else PresentationPriority.WARNING
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.SPEECH,
                    priority=prio,
                    message_id=msg_id,
                    text="Slow down",
                    direction=None,
                    intensity=0.7,
                    duration=0.8,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                )
            )
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.HAPTIC,
                    priority=prio,
                    message_id=msg_id,
                    text=None,
                    direction=None,
                    intensity=0.6,
                    duration=0.6,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                    haptic_pattern=HapticPattern.SLOW,
                )
            )
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.VISUAL,
                    priority=prio,
                    message_id=msg_id,
                    text=SpeechFormatter.format_visual_banner(action, decision.reason, risk_lvl),
                    direction=None,
                    intensity=0.7,
                    duration=0.5,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                )
            )
            return instructions

        # --------------------------------------------------------------------
        # 3. TURN_LEFT / TURN_RIGHT
        # --------------------------------------------------------------------
        if action in (NavigationAction.TURN_LEFT, NavigationAction.TURN_RIGHT):
            is_left = action == NavigationAction.TURN_LEFT
            sp_text = "Turn left" if is_left else "Turn right"
            h_pat = HapticPattern.LEFT if is_left else HapticPattern.RIGHT
            def_angle = -0.40 if is_left else 0.40
            cue_angle = heading if heading is not None else def_angle

            prio = PresentationPriority.URGENT if risk_lvl == RiskLevel.HIGH else PresentationPriority.WARNING

            # Speech
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.SPEECH,
                    priority=prio,
                    message_id=msg_id,
                    text=sp_text,
                    direction=cue_angle,
                    intensity=0.75,
                    duration=0.8,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                )
            )
            # Spatial Audio
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.SPATIAL_AUDIO,
                    priority=prio,
                    message_id=msg_id,
                    text=None,
                    direction=cue_angle,
                    intensity=0.80,
                    duration=0.6,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                )
            )
            # Haptic
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.HAPTIC,
                    priority=prio,
                    message_id=msg_id,
                    text=None,
                    direction=cue_angle,
                    intensity=0.70,
                    duration=0.6,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                    haptic_pattern=h_pat,
                )
            )
            # Visual
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.VISUAL,
                    priority=prio,
                    message_id=msg_id,
                    text=SpeechFormatter.format_visual_banner(action, decision.reason, risk_lvl),
                    direction=cue_angle,
                    intensity=0.75,
                    duration=0.5,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                )
            )
            return instructions

        # --------------------------------------------------------------------
        # 4. CAUTION
        # --------------------------------------------------------------------
        if action == NavigationAction.CAUTION:
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.SPEECH,
                    priority=PresentationPriority.CAUTION,
                    message_id=msg_id,
                    text="Caution ahead",
                    direction=heading,
                    intensity=0.5,
                    duration=0.8,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                )
            )
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.HAPTIC,
                    priority=PresentationPriority.CAUTION,
                    message_id=msg_id,
                    text=None,
                    direction=heading,
                    intensity=0.4,
                    duration=0.4,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                    haptic_pattern=HapticPattern.WARNING,
                )
            )
            instructions.append(
                PresentationInstruction(
                    channel=PresentationChannel.VISUAL,
                    priority=PresentationPriority.CAUTION,
                    message_id=msg_id,
                    text=SpeechFormatter.format_visual_banner(action, decision.reason, risk_lvl),
                    direction=heading,
                    intensity=0.5,
                    duration=0.5,
                    timestamp=t,
                    source_action=action,
                    source_risk_level=risk_lvl,
                    threat_id=threat_id,
                )
            )
            return instructions

        # --------------------------------------------------------------------
        # 5. CONTINUE
        # --------------------------------------------------------------------
        instructions.append(
            PresentationInstruction(
                channel=PresentationChannel.VISUAL,
                priority=PresentationPriority.INFO,
                message_id=msg_id,
                text="[SAFE] CONTINUE",
                direction=None,
                intensity=0.2,
                duration=0.5,
                timestamp=t,
                source_action=action,
                source_risk_level=risk_lvl,
                threat_id=threat_id,
            )
        )
        return instructions
