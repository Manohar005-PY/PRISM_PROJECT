"""Unit tests for CameraFrame, CameraSource, and MockCameraSource."""

import unittest

from app.runtime.realtime.camera import CameraConfig, CameraFrame, MockCameraSource


class TestCamera(unittest.TestCase):
    """Test suite for camera frame representation and sources."""

    def test_camera_frame_immutability(self) -> None:
        frame = CameraFrame(
            frame="dummy_pixels",
            timestamp=100.5,
            frame_id=1,
            width=640,
            height=480,
        )
        self.assertEqual(frame.timestamp, 100.5)
        self.assertEqual(frame.frame_id, 1)

    def test_mock_camera_source(self) -> None:
        source = MockCameraSource(CameraConfig(width=640, height=480))
        self.assertFalse(source.is_opened())
        self.assertTrue(source.open())
        self.assertTrue(source.is_opened())

        f1 = source.read_frame()
        self.assertIsNotNone(f1)
        self.assertEqual(f1.frame_id, 1)  # type: ignore[union-attr]

        source.release()
        self.assertFalse(source.is_opened())


if __name__ == "__main__":
    unittest.main()
