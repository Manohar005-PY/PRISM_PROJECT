"""Unit tests for MultiObjectTracker lifecycle and ID stability."""

import unittest

from app.core.types import Vector2
from app.perception.model import PhysicalObservation
from app.tracking.config import TrackingConfig
from app.tracking.model import TrackStatus
from app.tracking.tracker import MultiObjectTracker


class TestMultiObjectTracker(unittest.TestCase):
    """Test suite for MultiObjectTracker."""

    def setUp(self) -> None:
        self.config = TrackingConfig(confirmation_hits=2, max_missed_frames=3)
        self.tracker = MultiObjectTracker(config=self.config)

    def test_single_object_tracking_continuity(self) -> None:
        """Verify that a single object retains the same track_id across continuous observations."""
        track_ids = []
        for i in range(5):
            t = i * 0.1
            obs = PhysicalObservation(
                position=Vector2(0.0, 5.0 - i * 0.2),
                measurement_covariance=((0.04, 0.0), (0.0, 0.09)),
                confidence=0.9,
                timestamp=t,
                detection_id=f"det_{100 + i}",  # detection ID changes every frame!
                class_name="person",
            )
            result = self.tracker.update([obs], timestamp=t)
            self.assertEqual(len(result.tracked_observations), 1)
            track_ids.append(result.tracked_observations[0].track_id)

        # Track ID must remain 1 throughout despite changing detection IDs
        self.assertEqual(set(track_ids), {1})

    def test_two_objects_independent_track_ids(self) -> None:
        """Verify that two simultaneous objects receive distinct, persistent track IDs."""
        for i in range(4):
            t = i * 0.1
            obs1 = PhysicalObservation(
                position=Vector2(-1.0, 4.0),
                measurement_covariance=((0.04, 0.0), (0.0, 0.04)),
                confidence=0.9,
                timestamp=t,
                detection_id=f"d1_{i}",
                class_name="person",
            )
            obs2 = PhysicalObservation(
                position=Vector2(1.5, 6.0),
                measurement_covariance=((0.04, 0.0), (0.0, 0.04)),
                confidence=0.9,
                timestamp=t,
                detection_id=f"d2_{i}",
                class_name="chair",
            )
            result = self.tracker.update([obs1, obs2], timestamp=t)
            self.assertEqual(len(result.tracked_observations), 2)
            t_ids = {to.track_id for to in result.tracked_observations}
            self.assertEqual(t_ids, {1, 2})

    def test_track_id_monotonicity_and_no_recycling(self) -> None:
        """Deleted track IDs must never be recycled."""
        # 1. Create track 1
        obs1 = PhysicalObservation(
            position=Vector2(0.0, 3.0),
            measurement_covariance=((0.04, 0.0), (0.0, 0.04)),
            confidence=0.9,
            timestamp=0.0,
            detection_id="d1",
            class_name="person",
        )
        self.tracker.update([obs1], timestamp=0.0)

        # 2. Miss frames until track 1 is deleted
        for step in range(1, 6):
            self.tracker.update([], timestamp=step * 0.1)

        # Verify track 1 is deleted
        trk1 = self.tracker.get_track(1)
        self.assertIsNotNone(trk1)
        self.assertEqual(trk1.status, TrackStatus.DELETED)  # type: ignore[union-attr]

        # 3. New observation appears -> must get track_id = 2, NOT 1
        obs2 = PhysicalObservation(
            position=Vector2(0.0, 3.0),
            measurement_covariance=((0.04, 0.0), (0.0, 0.04)),
            confidence=0.9,
            timestamp=1.0,
            detection_id="d2",
            class_name="person",
        )
        res = self.tracker.update([obs2], timestamp=1.0)
        self.assertEqual(res.tracked_observations[0].track_id, 2)


if __name__ == "__main__":
    unittest.main()
