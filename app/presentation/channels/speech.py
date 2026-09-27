"""Text and TTS speech presentation channel adapter."""

from __future__ import annotations

from app.presentation.channels.base import SpeechChannel
from app.presentation.model import PresentationChannel, PresentationInstruction


class TextSpeechChannel(SpeechChannel):
    """Simple, platform-independent speech delivery logging utterances to a buffer or stdout."""

    def __init__(self, print_to_stdout: bool = False) -> None:
        self.print_to_stdout = print_to_stdout
        self.utterance_history: list[str] = []

    def channel_type(self) -> PresentationChannel:
        return PresentationChannel.SPEECH

    def is_available(self) -> bool:
        return True

    def deliver(self, instruction: PresentationInstruction) -> bool:
        if instruction.channel != PresentationChannel.SPEECH or instruction.text is None:
            return False

        text = instruction.text.strip()
        if not text:
            return True

        self.utterance_history.append(text)
        if self.print_to_stdout:
            print(f"[SPEECH] {text}")
        return True
