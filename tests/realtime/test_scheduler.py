"""Unit tests for RealtimeScheduler and stale-frame validation."""

import unittest

from app.runtime.realtime.camera import CameraFrame
from app.runtime.realtime.config import RealtimeConfig
from app.runtime.realtime.scheduler import RealtimeScheduler


class TestRealtimeScheduler(unittest.TestCase):
    """Test suite for RealtimeScheduler."""

    def test_stale_frame_detection(self) -> None:
        cfg = RealtimeConfig(max_frame_age=0.20)
        scheduler = RealtimeScheduler(cfg)

        # Fresh frame (captured at t=1.0, current time t=1.1 => age=0.1s < 0.2s)
        fresh_frame = CameraFrame("pixels", timestamp=1.0, frame_id=1, width=640, height=480)
        self.assertFalse(scheduler.is_frame_stale(fresh_frame, current_time=1.1))

        # Stale frame (captured at t=1.0, current time t=1.35 => age=0.35s > 0.2s)
        stale_frame = CameraFrame("pixels", timestamp=1.0, frame_id=1, width=640, height=480)
        self.assertTrue(scheduler.is_frame_stale(stale_frame, current_time=1.35))


if __name__ == "__main__":
    unittest.main()
