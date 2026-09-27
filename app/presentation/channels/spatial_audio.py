"""Spatial audio presentation channel adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.presentation.channels.base import SpatialAudioChannel
from app.presentation.model import PresentationChannel, PresentationInstruction, PresentationPriority


@dataclass(frozen=True)
class SpatialAudioCueRecord:
    """Delivered spatial audio record."""
    direction_rad: float
    intensity: float
    duration: float
    priority: PresentationPriority
    timestamp: float


class SpatialAudioChannelAdapter(SpatialAudioChannel):
    """Platform-independent spatial audio channel recording spatialized cues."""

    def __init__(self) -> None:
        self.cues: list[SpatialAudioCueRecord] = []

    def channel_type(self) -> PresentationChannel:
        return PresentationChannel.SPATIAL_AUDIO

    def is_available(self) -> bool:
        return True

    def deliver(self, instruction: PresentationInstruction) -> bool:
        if instruction.channel != PresentationChannel.SPATIAL_AUDIO:
            return False

        angle = instruction.direction if instruction.direction is not None else 0.0
        cue = SpatialAudioCueRecord(
            direction_rad=angle,
            intensity=instruction.intensity,
            duration=instruction.duration,
            priority=instruction.priority,
            timestamp=instruction.timestamp,
        )
        self.cues.append(cue)
        return True
