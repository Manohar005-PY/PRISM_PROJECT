"""Unit tests for RealtimePerformanceTracker, percentiles, and throughput metrics."""

import unittest

from app.runtime.realtime.metrics import (
    RealtimePerformanceTracker,
    compute_percentile,
)


class TestRealtimeMetrics(unittest.TestCase):
    """Test suite for metrics and percentile calculations."""

    def test_compute_percentile(self) -> None:
        data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        self.assertAlmostEqual(compute_percentile(data, 50.0), 55.0)
        self.assertAlmostEqual(compute_percentile(data, 95.0), 95.5)

    def test_tracker_accumulation(self) -> None:
        tracker = RealtimePerformanceTracker(window_size=10)
        tracker.record_capture(1.0)
        tracker.record_capture(1.033)
        tracker.record_capture(1.066)

        tracker.record_processed(15.0, 10.0, 1.050)
        tracker.record_processed(16.0, 11.0, 1.080)
        tracker.record_drop()

        m = tracker.compute_metrics()
        self.assertGreater(m.mean_latency_ms, 0.0)
        self.assertEqual(m.dropped_frames, 1)
        self.assertGreater(m.drop_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
