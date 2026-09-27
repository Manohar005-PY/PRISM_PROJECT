"""Unit tests for CameraConfig, InferenceConfig, and RealtimeConfig."""

import unittest

from app.runtime.realtime.config import CameraConfig, InferenceConfig, RealtimeConfig


class TestRealtimeConfig(unittest.TestCase):
    """Test suite for realtime configuration validation."""

    def test_camera_config_defaults(self) -> None:
        cfg = CameraConfig()
        self.assertEqual(cfg.width, 640)
        self.assertEqual(cfg.height, 480)
        self.assertEqual(cfg.target_fps, 30.0)

    def test_inference_config_defaults(self) -> None:
        cfg = InferenceConfig()
        self.assertEqual(cfg.confidence_threshold, 0.40)
        self.assertEqual(cfg.image_size, 640)

    def test_realtime_config_invariants(self) -> None:
        cfg = RealtimeConfig()
        self.assertEqual(cfg.max_frame_age, 0.20)
        self.assertEqual(cfg.max_processing_latency, 0.10)


if __name__ == "__main__":
    unittest.main()
