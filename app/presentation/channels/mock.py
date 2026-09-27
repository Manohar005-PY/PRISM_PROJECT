"""Deterministic mock presentation channels for test automation and validation."""

from __future__ import annotations

from app.presentation.model import PresentationChannel, PresentationInstruction


class BaseMockChannel:
    """Base mock channel recording delivery history with optional simulated failures."""

    def __init__(self, channel: PresentationChannel, available: bool = True) -> None:
        self._channel = channel
        self._available = available
        self.delivered_instructions: list[PresentationInstruction] = []
        self.fail_on_deliver: bool = False

    def channel_type(self) -> PresentationChannel:
        return self._channel

    def is_available(self) -> bool:
        return self._available

    def set_available(self, available: bool) -> None:
        self._available = available

    def deliver(self, instruction: PresentationInstruction) -> bool:
        if not self._available or self.fail_on_deliver:
            return False
        self.delivered_instructions.append(instruction)
        return True

    def clear(self) -> None:
        self.delivered_instructions.clear()


class MockSpeechChannel(BaseMockChannel):
    """Mock speech channel."""
    def __init__(self, available: bool = True) -> None:
        super().__init__(PresentationChannel.SPEECH, available)


class MockSpatialAudioChannel(BaseMockChannel):
    """Mock spatial audio channel."""
    def __init__(self, available: bool = True) -> None:
        super().__init__(PresentationChannel.SPATIAL_AUDIO, available)


class MockHapticChannel(BaseMockChannel):
    """Mock haptic channel."""
    def __init__(self, available: bool = True) -> None:
        super().__init__(PresentationChannel.HAPTIC, available)


class MockVisualChannel(BaseMockChannel):
    """Mock visual channel."""
    def __init__(self, available: bool = True) -> None:
        super().__init__(PresentationChannel.VISUAL, available)
