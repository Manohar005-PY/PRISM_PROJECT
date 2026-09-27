"""Unit tests for TemporalRiskTracker exponential smoothing and track pruning."""

import unittest

from app.risk.temporal import TemporalRiskTracker


class TestTemporalRiskTracker(unittest.TestCase):
    """Test suite for TemporalRiskTracker."""

    def test_first_observation_initialization(self) -> None:
        tracker = TemporalRiskTracker(default_alpha=0.5)
        # First update initializes exact value
        score = tracker.update_score(object_id=1, raw_score=0.8)
        self.assertEqual(score, 0.8)

    def test_exponential_smoothing(self) -> None:
        tracker = TemporalRiskTracker(default_alpha=0.5)
        tracker.update_score(1, 0.8)  # Prev = 0.8
        # Step 2: raw = 0.4 => smooth = 0.5 * 0.4 + 0.5 * 0.8 = 0.6
        score2 = tracker.update_score(1, 0.4)
        self.assertAlmostEqual(score2, 0.6)

    def test_prune_inactive(self) -> None:
        tracker = TemporalRiskTracker()
        tracker.update_score(1, 0.5)
        tracker.update_score(2, 0.7)

        tracker.prune_inactive(active_object_ids={1})
        self.assertIn(1, tracker.smoothed_scores)
        self.assertNotIn(2, tracker.smoothed_scores)

    def test_reset(self) -> None:
        tracker = TemporalRiskTracker()
        tracker.update_score(1, 0.5)
        tracker.reset()
        self.assertEqual(len(tracker.smoothed_scores), 0)


if __name__ == "__main__":
    unittest.main()
