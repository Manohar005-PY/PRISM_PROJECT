"""Unit tests for HealthMonitor and RuntimeStatus transitions."""

import unittest

from app.runtime.config import RuntimeConfig
from app.runtime.health import HealthMonitor, RuntimeStatus


class TestRuntimeHealth(unittest.TestCase):
    """Test suite for runtime health monitoring."""

    def test_healthy_transition(self) -> None:
        cfg = RuntimeConfig(maximum_consecutive_failures=3)
        monitor = HealthMonitor(cfg)
        monitor.record_success()

        health = monitor.assess_health(active_tracks=2, stale_tracks=0, average_latency_ms=10.0, current_fps=100.0)
        self.assertEqual(health.status, RuntimeStatus.HEALTHY)
        self.assertTrue(health.is_healthy())

    def test_degraded_due_to_stale_tracks(self) -> None:
        cfg = RuntimeConfig()
        monitor = HealthMonitor(cfg)
        monitor.record_success()

        health = monitor.assess_health(active_tracks=2, stale_tracks=1, average_latency_ms=10.0, current_fps=100.0)
        self.assertEqual(health.status, RuntimeStatus.DEGRADED)
        self.assertFalse(health.is_healthy())

    def test_failed_due_to_consecutive_failures(self) -> None:
        cfg = RuntimeConfig(maximum_consecutive_failures=3)
        monitor = HealthMonitor(cfg)
        monitor.record_failure("error 1")
        monitor.record_failure("error 2")
        monitor.record_failure("error 3")

        health = monitor.assess_health(active_tracks=0, stale_tracks=0, average_latency_ms=0.0, current_fps=0.0)
        self.assertEqual(health.status, RuntimeStatus.FAILED)
        self.assertEqual(health.consecutive_failures, 3)


if __name__ == "__main__":
    unittest.main()
