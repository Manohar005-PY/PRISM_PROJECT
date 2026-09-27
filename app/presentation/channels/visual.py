"""Visual HUD and display overlay presentation channel adapter."""

from __future__ import annotations

from dataclasses import dataclass

from app.presentation.channels.base import VisualChannel
from app.presentation.model import PresentationChannel, PresentationInstruction, PresentationPriority


@dataclass(frozen=True)
class VisualHUDRecord:
    """Delivered visual display record."""
    banner_text: str
    priority: PresentationPriority
    timestamp: float


class VisualChannelAdapter(VisualChannel):
    """Visual display and HUD text banner adapter."""

    def __init__(self) -> None:
        self.hud_records: list[VisualHUDRecord] = []
        self.latest_banner: str | None = None

    def channel_type(self) -> PresentationChannel:
        return PresentationChannel.VISUAL

    def is_available(self) -> bool:
        return True

    def deliver(self, instruction: PresentationInstruction) -> bool:
        if instruction.channel != PresentationChannel.VISUAL or instruction.text is None:
            return False

        rec = VisualHUDRecord(
            banner_text=instruction.text,
            priority=instruction.priority,
            timestamp=instruction.timestamp,
        )
        self.hud_records.append(rec)
        self.latest_banner = instruction.text
        return True
