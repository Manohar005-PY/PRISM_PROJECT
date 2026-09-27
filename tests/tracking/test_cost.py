"""Unit tests for cost matrix generation and gating rejection."""

import unittest

from app.core.types import ObjectState, Vector2
from app.perception.model import PhysicalObservation
from app.tracking.config import TrackingConfig
from app.tracking.cost import INF_COST, build_cost_matrix
from app.tracking.model import Track, TrackStatus


class TestTrackingCost(unittest.TestCase):
    """Test suite for cost matrix construction."""

    def setUp(self) -> None:
        self.config = TrackingConfig(gating_threshold=9.21, class_aware=True)
        self.trk1 = Track(1, "person", TrackStatus.CONFIRMED, 0.0, 0.0)
        self.trk2 = Track(2, "chair", TrackStatus.CONFIRMED, 0.0, 0.0)

        self.state1 = ObjectState(
            object_id=1,
            class_id="person",
            position=Vector2(0.0, 3.0),
            velocity=Vector2(0.0, 0.0),
            timestamp=0.0,
            covariance=((0.1, 0.0, 0.0, 0.0), (0.0, 0.1, 0.0, 0.0), (0.0, 0.0, 0.1, 0.0), (0.0, 0.0, 0.0, 0.1)),
        )
        self.state2 = ObjectState(
            object_id=2,
            class_id="chair",
            position=Vector2(2.0, 5.0),
            velocity=Vector2(0.0, 0.0),
            timestamp=0.0,
            covariance=((0.1, 0.0, 0.0, 0.0), (0.0, 0.1, 0.0, 0.0), (0.0, 0.0, 0.1, 0.0), (0.0, 0.0, 0.0, 0.1)),
        )

    def test_cost_matrix_matching_and_class_gating(self) -> None:
        obs1 = PhysicalObservation(
            position=Vector2(0.05, 3.05),
            measurement_covariance=((0.04, 0.0), (0.0, 0.04)),
            confidence=0.9,
            timestamp=0.1,
            detection_id="d1",
            class_name="person",
        )
        obs2 = PhysicalObservation(
            position=Vector2(2.05, 5.05),
            measurement_covariance=((0.04, 0.0), (0.0, 0.04)),
            confidence=0.9,
            timestamp=0.1,
            detection_id="d2",
            class_name="chair",
        )

        cost_mat = build_cost_matrix(
            tracks=[self.trk1, self.trk2],
            predicted_states={1: self.state1, 2: self.state2},
            observations=[obs1, obs2],
            config=self.config,
        )

        # Track 1 (person) matching Obs 1 (person) -> low cost
        self.assertLess(cost_mat[0][0], 5.0)
        # Track 1 (person) matching Obs 2 (chair) -> INF (class mismatch)
        self.assertGreaterEqual(cost_mat[0][1], INF_COST / 2)

        # Track 2 (chair) matching Obs 1 (person) -> INF (class mismatch)
        self.assertGreaterEqual(cost_mat[1][0], INF_COST / 2)
        # Track 2 (chair) matching Obs 2 (chair) -> low cost
        self.assertLess(cost_mat[1][1], 5.0)


if __name__ == "__main__":
    unittest.main()
