"""Unit tests for RuntimeConfig validation and defaults."""

import unittest

from app.decision.model import NavigationAction
from app.runtime.config import RuntimeConfig


class TestRuntimeConfig(unittest.TestCase):
    """Test suite for RuntimeConfig."""

    def test_default_config(self) -> None:
        cfg = RuntimeConfig()
        self.assertEqual(cfg.target_fps, 30.0)
        self.assertEqual(cfg.max_frame_processing_time, 0.100)
        self.assertEqual(cfg.stale_observation_timeout, 1.0)
        self.assertEqual(cfg.fail_safe_action, NavigationAction.STOP)

    def test_invalid_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            RuntimeConfig(target_fps=-5.0)
        with self.assertRaises(ValueError):
            RuntimeConfig(max_frame_processing_time=0.0)
        with self.assertRaises(TypeError):
            RuntimeConfig(fail_safe_action="INVALID")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
