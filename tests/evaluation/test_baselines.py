"""Unit tests for reactive baselines (Distance & TTC)."""

import unittest

from app.core.types import Vector2
from app.decision.model import NavigationAction
from app.perception.model import PhysicalObservation
from evaluation.baselines import DistanceBaseline, TTCBaseline


class TestBaselines(unittest.TestCase):
    """Test suite for reactive baselines."""

    def test_distance_baseline_thresholds(self) -> None:
        base = DistanceBaseline(stop_threshold=2.0, caution_threshold=4.0)

        # Far away (Z=5.0) -> CONTINUE
        obs_far = [PhysicalObservation(Vector2(0.0, 5.0), ((0.04, 0.0), (0.0, 0.04)), 0.9, 0.1, "d1", "person")]
        risk, dec = base.process(obs_far, 0.1)
        self.assertEqual(dec.action, NavigationAction.CONTINUE)

        # Close (Z=1.5) -> STOP
        obs_close = [PhysicalObservation(Vector2(0.0, 1.5), ((0.04, 0.0), (0.0, 0.04)), 0.9, 0.2, "d1", "person")]
        risk, dec = base.process(obs_close, 0.2)
        self.assertEqual(dec.action, NavigationAction.STOP)

    def test_ttc_baseline_velocity_response(self) -> None:
        base = TTCBaseline(ttc_threshold=2.5)

        # Frame 1: Z=4.0 at t=0.0
        obs1 = [PhysicalObservation(Vector2(0.0, 4.0), ((0.04, 0.0), (0.0, 0.04)), 0.9, 0.0, "d1", "person")]
        base.process(obs1, 0.0)

        # Frame 2: Z=3.8 at t=0.1 (approaching at vz = -2.0 m/s => TTC = 3.8/2.0 = 1.9s < 2.5s)
        obs2 = [PhysicalObservation(Vector2(0.0, 3.8), ((0.04, 0.0), (0.0, 0.04)), 0.9, 0.1, "d1", "person")]
        risk, dec = base.process(obs2, 0.1)
        self.assertEqual(dec.action, NavigationAction.STOP)


if __name__ == "__main__":
    unittest.main()
