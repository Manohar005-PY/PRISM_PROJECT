"""Multimodal presentation channel adapters and protocols."""

from app.presentation.channels.base import (
    HapticChannel,
    PresentationChannelAdapter,
    SpatialAudioChannel,
    SpeechChannel,
    VisualChannel,
)
from app.presentation.channels.haptic import HapticChannelAdapter
from app.presentation.channels.mock import (
    MockHapticChannel,
    MockSpatialAudioChannel,
    MockSpeechChannel,
    MockVisualChannel,
)
from app.presentation.channels.spatial_audio import SpatialAudioChannelAdapter
from app.presentation.channels.speech import TextSpeechChannel
from app.presentation.channels.visual import VisualChannelAdapter

__all__ = [
    "PresentationChannelAdapter",
    "SpeechChannel",
    "SpatialAudioChannel",
    "HapticChannel",
    "VisualChannel",
    "MockSpeechChannel",
    "MockSpatialAudioChannel",
    "MockHapticChannel",
    "MockVisualChannel",
    "TextSpeechChannel",
    "SpatialAudioChannelAdapter",
    "HapticChannelAdapter",
    "VisualChannelAdapter",
]
