"""Unit tests for StageTiming and RollingTimingProfiler."""

import unittest

from app.runtime.timing import RollingTimingProfiler, StageTiming


class TestRuntimeTiming(unittest.TestCase):
    """Test suite for timing instrumentation."""

    def test_stage_timing_invariants(self) -> None:
        timing = StageTiming(
            perception_ms=2.0,
            tracking_ms=1.5,
            estimation_ms=0.5,
            prediction_ms=0.5,
            risk_ms=0.5,
            decision_ms=0.5,
            total_ms=5.5,
            effective_fps=181.8,
        )
        self.assertEqual(timing.total_ms, 5.5)
        self.assertAlmostEqual(timing.effective_fps, 181.8)

    def test_rolling_profiler_statistics(self) -> None:
        profiler = RollingTimingProfiler(window_size=3)
        profiler.record(StageTiming(total_ms=10.0))
        profiler.record(StageTiming(total_ms=20.0))
        profiler.record(StageTiming(total_ms=30.0))

        self.assertEqual(profiler.sample_count, 3)
        self.assertAlmostEqual(profiler.average_latency_ms, 20.0)
        self.assertAlmostEqual(profiler.min_latency_ms, 10.0)
        self.assertAlmostEqual(profiler.max_latency_ms, 30.0)
        self.assertAlmostEqual(profiler.average_fps, 50.0)


if __name__ == "__main__":
    unittest.main()
