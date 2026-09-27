"""Presentation dispatcher routing instructions to channels with complete failure isolation."""

from __future__ import annotations

from typing import Sequence

from app.presentation.channels.base import PresentationChannelAdapter
from app.presentation.model import PresentationChannel, PresentationInstruction


class PresentationDispatcher:
    """Dispatches PresentationInstructions to registered channels, isolating all delivery failures."""

    def __init__(self, channels: Sequence[PresentationChannelAdapter] | None = None) -> None:
        self._channels: dict[PresentationChannel, PresentationChannelAdapter] = {}
        self.delivered_count: int = 0
        self.failure_count: int = 0
        self.channel_failure_events: list[tuple[PresentationChannel, str]] = []

        if channels:
            for ch in channels:
                self.register_channel(ch)

    def register_channel(self, channel: PresentationChannelAdapter) -> None:
        """Register a delivery channel adapter."""
        self._channels[channel.channel_type()] = channel

    def get_channel(self, channel_type: PresentationChannel) -> PresentationChannelAdapter | None:
        """Retrieve registered channel adapter by type."""
        return self._channels.get(channel_type)

    def dispatch(
        self,
        instructions: Sequence[PresentationInstruction],
    ) -> dict[PresentationChannel, bool]:
        """Deliver each instruction to its target channel adapter.

        Failure in one channel will never prevent delivery to subsequent channels.

        Args:
            instructions: Sequence of PresentationInstruction instances.

        Returns:
            Dictionary mapping PresentationChannel to delivery success status.
        """
        results: dict[PresentationChannel, bool] = {}

        for inst in instructions:
            ch_type = inst.channel
            adapter = self._channels.get(ch_type)

            if adapter is None or not adapter.is_available():
                results[ch_type] = False
                self.failure_count += 1
                self.channel_failure_events.append((ch_type, "channel_unavailable"))
                continue

            try:
                success = adapter.deliver(inst)
                results[ch_type] = success
                if success:
                    self.delivered_count += 1
                else:
                    self.failure_count += 1
                    self.channel_failure_events.append((ch_type, "delivery_rejected"))
            except Exception as ex:
                results[ch_type] = False
                self.failure_count += 1
                self.channel_failure_events.append((ch_type, str(ex)))

        return results
