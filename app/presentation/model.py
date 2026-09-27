"""Domain models, enums, instructions, and configuration for PRISM Presentation subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

from app.core.enums import RiskLevel
from app.decision.model import NavigationAction


class PresentationChannel(str, Enum):
    """Available multimodal presentation delivery channels."""

    SPEECH = "speech"
    SPATIAL_AUDIO = "spatial_audio"
    HAPTIC = "haptic"
    VISUAL = "visual"


class PresentationPriority(str, Enum):
    """Precedence priority for presentation instructions."""

    INFO = "info"
    CAUTION = "caution"
    WARNING = "warning"
    URGENT = "urgent"
    EMERGENCY = "emergency"


def get_priority_level(priority: PresentationPriority) -> int:
    """Return integer priority rank (higher = greater urgency)."""
    mapping = {
        PresentationPriority.INFO: 1,
        PresentationPriority.CAUTION: 2,
        PresentationPriority.WARNING: 3,
        PresentationPriority.URGENT: 4,
        PresentationPriority.EMERGENCY: 5,
    }
    return mapping.get(priority, 1)


class HapticPattern(str, Enum):
    """Discrete tactile vibration patterns."""

    LEFT = "left"
    RIGHT = "right"
    SLOW = "slow"
    WARNING = "warning"
    STOP = "stop"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class PresentationConfig:
    """Configuration parameters and cooldown intervals for multimodal presentation."""

    speech_cooldown: float = 1.5
    spatial_audio_cooldown: float = 1.0
    haptic_cooldown: float = 0.8
    repeated_action_cooldown: float = 2.0
    emergency_bypass: bool = True
    risk_escalation_bypass: bool = True

    def __post_init__(self) -> None:
        for name, val in [
            ("speech_cooldown", self.speech_cooldown),
            ("spatial_audio_cooldown", self.spatial_audio_cooldown),
            ("haptic_cooldown", self.haptic_cooldown),
            ("repeated_action_cooldown", self.repeated_action_cooldown),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be non-negative finite float, got {val!r}")


@dataclass(frozen=True)
class PresentationInstruction:
    """Immutable, machine-readable presentation instruction targeted to a specific channel.

    Attributes:
        channel: Target PresentationChannel (speech, spatial audio, haptic, visual).
        priority: Urgency priority level.
        message_id: Semantic identifier for deduplication and logging.
        text: Optional speech utterance or visual text banner.
        direction: Optional continuous ground-plane heading angle in radians (+X right, -X left).
        intensity: Normalized cue intensity in [0.0, 1.0].
        duration: Active cue duration in seconds.
        timestamp: Creation timestamp in seconds.
        source_action: Upstream NavigationAction that generated this instruction.
        source_risk_level: Upstream RiskLevel that justified the instruction.
        threat_id: ID of the primary obstacle prompting the cue (or None).
        haptic_pattern: Optional tactile pattern if channel is HAPTIC.
    """

    channel: PresentationChannel
    priority: PresentationPriority
    message_id: str
    text: str | None
    direction: float | None
    intensity: float
    duration: float
    timestamp: float
    source_action: NavigationAction
    source_risk_level: RiskLevel
    threat_id: int | None = None
    haptic_pattern: HapticPattern | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.channel, PresentationChannel):
            raise TypeError(f"channel must be PresentationChannel, got {type(self.channel)}")
        if not isinstance(self.priority, PresentationPriority):
            raise TypeError(f"priority must be PresentationPriority, got {type(self.priority)}")
        if not isinstance(self.message_id, str) or not self.message_id.strip():
            raise ValueError(f"message_id must be non-empty string, got {self.message_id!r}")
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError(f"text must be string or None, got {type(self.text)}")
        if self.direction is not None:
            if not isinstance(self.direction, (int, float)) or not math.isfinite(self.direction):
                raise ValueError(f"direction must be finite float or None, got {self.direction!r}")
            object.__setattr__(self, "direction", float(self.direction))
        if not isinstance(self.intensity, (int, float)) or not math.isfinite(self.intensity) or not (0.0 <= self.intensity <= 1.0):
            raise ValueError(f"intensity must be in [0.0, 1.0], got {self.intensity!r}")
        if not isinstance(self.duration, (int, float)) or not math.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError(f"duration must be positive float, got {self.duration!r}")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be finite float, got {self.timestamp!r}")
        if not isinstance(self.source_action, NavigationAction):
            raise TypeError(f"source_action must be NavigationAction, got {type(self.source_action)}")
        if not isinstance(self.source_risk_level, RiskLevel):
            raise TypeError(f"source_risk_level must be RiskLevel, got {type(self.source_risk_level)}")
        if self.threat_id is not None and not isinstance(self.threat_id, int):
            raise TypeError(f"threat_id must be int or None, got {type(self.threat_id)}")
        if self.haptic_pattern is not None and not isinstance(self.haptic_pattern, HapticPattern):
            raise TypeError(f"haptic_pattern must be HapticPattern or None, got {type(self.haptic_pattern)}")

        object.__setattr__(self, "intensity", float(self.intensity))
        object.__setattr__(self, "duration", float(self.duration))
        object.__setattr__(self, "timestamp", float(self.timestamp))
