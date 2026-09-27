"""Real-time frame scheduler, rate-limiting, and frame freshness validation."""

from __future__ import annotations

import time

from app.runtime.realtime.camera import CameraFrame
from app.runtime.realtime.config import RealtimeConfig


class RealtimeScheduler:
    """Manages frame rate throttling and frame age freshness verification."""

    def __init__(self, config: RealtimeConfig | None = None) -> None:
        self.config: RealtimeConfig = config if config is not None else RealtimeConfig()
        self._target_interval: float = 1.0 / self.config.camera.target_fps
        self._last_processed_time: float | None = None

    def is_frame_stale(self, frame: CameraFrame, current_time: float) -> bool:
        """Check if frame acquisition timestamp exceeds allowable freshness budget."""
        age = current_time - frame.timestamp
        return age > self.config.max_frame_age

    def pace_frame_rate(self) -> None:
        """Throttle processing loop to adhere to configured target FPS."""
        now = time.monotonic()
        if self._last_processed_time is not None:
            elapsed = now - self._last_processed_time
            sleep_time = self._target_interval - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)
        self._last_processed_time = time.monotonic()
