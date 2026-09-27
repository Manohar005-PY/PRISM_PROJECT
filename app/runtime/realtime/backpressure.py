"""Backpressure management and latest-frame-wins bounded buffer."""

from __future__ import annotations

from collections import deque
import threading
from typing import Generic, TypeVar

from app.runtime.realtime.camera import CameraFrame

T = TypeVar("T")


class LatestFrameBuffer:
    """Thread-safe bounded frame buffer implementing a latest-frame-wins dropping policy.

    Rationale:
        For assistive collision prediction and avoidance, processing a stale frame from
        several hundred milliseconds ago induces lethal latency. Discarding older waiting
        frames in favor of the most recent frame ensures physical decisions reflect current reality.
    """

    def __init__(self, maxsize: int = 1) -> None:
        if maxsize <= 0:
            raise ValueError(f"maxsize must be positive int, got {maxsize!r}")
        self.maxsize: int = maxsize
        self._queue: deque[CameraFrame] = deque(maxlen=maxsize)
        self._lock: threading.Lock = threading.Lock()
        self._dropped_count: int = 0

    def put(self, frame: CameraFrame) -> None:
        """Push a newly captured frame. If full, discards oldest frame and increments drop count."""
        with self._lock:
            if len(self._queue) == self.maxsize:
                self._dropped_count += 1
            self._queue.append(frame)

    def get(self) -> CameraFrame | None:
        """Retrieve and remove the latest frame from the buffer, or None if empty."""
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            return None

    def peek(self) -> CameraFrame | None:
        """Inspect the current frame in buffer without removing it."""
        with self._lock:
            if self._queue:
                return self._queue[-1]
            return None

    @property
    def dropped_count(self) -> int:
        """Total number of frames dropped due to processing backpressure."""
        with self._lock:
            return self._dropped_count

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
            self._dropped_count = 0
