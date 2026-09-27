"""Unit tests for tracking evaluation metrics (continuity, fragmentation, ID switches, latency, IDF1)."""

import unittest

from app.tracking.metrics import (
    association_accuracy,
    count_id_switches,
    identity_f1_score,
    track_confirmation_latency,
    track_continuity,
    track_fragmentation,
    track_precision,
    track_recall,
)


class TestTrackingMetrics(unittest.TestCase):
    """Test suite for tracking performance metrics."""

    def test_track_continuity(self) -> None:
        self.assertAlmostEqual(track_continuity(10, 9), 90.0)
        self.assertAlmostEqual(track_continuity(0, 0), 0.0)

    def test_track_fragmentation(self) -> None:
        gt_map = {1: {1}, 2: {2, 3}}
        self.assertAlmostEqual(track_fragmentation(gt_map), 1.5)

    def test_count_id_switches(self) -> None:
        seq = [1, 1, 1, 2, 2, 3]
        self.assertEqual(count_id_switches(seq), 2)

        seq_stable = [1, 1, None, 1, 1]
        self.assertEqual(count_id_switches(seq_stable), 0)

    def test_precision_and_recall(self) -> None:
        self.assertAlmostEqual(track_precision(18, 20), 90.0)
        self.assertAlmostEqual(track_recall(18, 20), 90.0)
        self.assertAlmostEqual(association_accuracy(19, 20), 95.0)

    def test_confirmation_latency_and_idf1(self) -> None:
        # Created frame 1, confirmed frame 3 -> 3 frames
        self.assertEqual(track_confirmation_latency(1, 3), 3)

        # IDF1: 2*IDTP / (2*IDTP + IDFP + IDFN)
        # IDTP = 18, IDFP = 2, IDFN = 2 => 36 / (36 + 4) = 36/40 = 90.0%
        self.assertAlmostEqual(identity_f1_score(18, 2, 2), 90.0)


if __name__ == "__main__":
    unittest.main()
