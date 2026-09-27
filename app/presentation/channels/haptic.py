"""Tactile and haptic vibration presentation channel adapter."""

from __future__ import annotations

from dataclasses import dataclass

from app.presentation.channels.base import HapticChannel
from app.presentation.model import HapticPattern, PresentationChannel, PresentationInstruction


@dataclass(frozen=True)
class HapticActuationRecord:
    """Delivered tactile actuation record."""
    pattern: HapticPattern
    intensity: float
    duration: float
    timestamp: float


class HapticChannelAdapter(HapticChannel):
    """Platform-independent tactile vibration driver adapter."""

    def __init__(self) -> None:
        self.actuations: list[HapticActuationRecord] = []

    def channel_type(self) -> PresentationChannel:
        return PresentationChannel.HAPTIC

    def is_available(self) -> bool:
        return True

    def deliver(self, instruction: PresentationInstruction) -> bool:
        if instruction.channel != PresentationChannel.HAPTIC or instruction.haptic_pattern is None:
            return False

        rec = HapticActuationRecord(
            pattern=instruction.haptic_pattern,
            intensity=instruction.intensity,
            duration=instruction.duration,
            timestamp=instruction.timestamp,
        )
        self.actuations.append(rec)
        return True
