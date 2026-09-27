"""Unit tests for LatestFrameBuffer backpressure and dropping policy."""

import unittest

from app.runtime.realtime.backpressure import LatestFrameBuffer
from app.runtime.realtime.camera import CameraFrame


class TestBackpressure(unittest.TestCase):
    """Test suite for bounded frame buffer and latest-frame-wins dropping policy."""

    def test_latest_frame_wins_bounded_queue(self) -> None:
        buf = LatestFrameBuffer(maxsize=1)
        self.assertTrue(buf.is_empty)

        f1 = CameraFrame("frame1", 1.0, 1, 640, 480)
        f2 = CameraFrame("frame2", 1.033, 2, 640, 480)
        f3 = CameraFrame("frame3", 1.066, 3, 640, 480)

        buf.put(f1)
        self.assertEqual(buf.dropped_count, 0)

        buf.put(f2)  # Overwrites f1 -> dropped_count = 1
        self.assertEqual(buf.dropped_count, 1)

        buf.put(f3)  # Overwrites f2 -> dropped_count = 2
        self.assertEqual(buf.dropped_count, 2)

        # Buffer must deliver latest frame (f3)
        got = buf.get()
        self.assertIsNotNone(got)
        self.assertEqual(got.frame_id, 3)  # type: ignore[union-attr]
        self.assertTrue(buf.is_empty)


if __name__ == "__main__":
    unittest.main()
