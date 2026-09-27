"""Base protocols and interfaces for presentation channel adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.presentation.model import PresentationChannel, PresentationInstruction


@runtime_checkable
class PresentationChannelAdapter(Protocol):
    """Generic protocol implemented by all multimodal delivery channel adapters."""

    def channel_type(self) -> PresentationChannel: ...
    def is_available(self) -> bool: ...
    def deliver(self, instruction: PresentationInstruction) -> bool: ...


@runtime_checkable
class SpeechChannel(PresentationChannelAdapter, Protocol):
    """Specialized protocol for speech / voice feedback."""
    def channel_type(self) -> PresentationChannel:
        return PresentationChannel.SPEECH


@runtime_checkable
class SpatialAudioChannel(PresentationChannelAdapter, Protocol):
    """Specialized protocol for spatialized 3D audio beacons."""
    def channel_type(self) -> PresentationChannel:
        return PresentationChannel.SPATIAL_AUDIO


@runtime_checkable
class HapticChannel(PresentationChannelAdapter, Protocol):
    """Specialized protocol for tactile haptic vibrations."""
    def channel_type(self) -> PresentationChannel:
        return PresentationChannel.HAPTIC


@runtime_checkable
class VisualChannel(PresentationChannelAdapter, Protocol):
    """Specialized protocol for visual HUD and display overlays."""
    def channel_type(self) -> PresentationChannel:
        return PresentationChannel.VISUAL
