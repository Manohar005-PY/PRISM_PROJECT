"""Unit tests for Track lifecycle transitions (TENTATIVE -> CONFIRMED -> LOST -> DELETED)."""

import unittest

from app.core.types import Vector2
from app.perception.model import PhysicalObservation
from app.tracking.config import TrackingConfig
from app.tracking.model import TrackStatus
from app.tracking.tracker import MultiObjectTracker


class TestTrackLifecycle(unittest.TestCase):
    """Test suite for Track lifecycle state machine."""

    def setUp(self) -> None:
        # Require 3 consecutive hits for confirmation; 3 missed frames before deletion
        self.config = TrackingConfig(confirmation_hits=3, max_missed_frames=3, tentative_max_missed=1)
        self.tracker = MultiObjectTracker(config=self.config)

    def test_confirmation_hits_transition(self) -> None:
        """Track starts TENTATIVE and becomes CONFIRMED exactly upon 3rd hit."""
        obs = lambda t: PhysicalObservation(
            position=Vector2(0.0, 4.0),
            measurement_covariance=((0.04, 0.0), (0.0, 0.04)),
            confidence=0.9,
            timestamp=t,
            detection_id="d1",
            class_name="person",
        )

        # Hit 1 -> TENTATIVE
        self.tracker.update([obs(0.0)], timestamp=0.0)
        self.assertEqual(self.tracker.get_track(1).status, TrackStatus.TENTATIVE)  # type: ignore[union-attr]

        # Hit 2 -> TENTATIVE
        self.tracker.update([obs(0.1)], timestamp=0.1)
        self.assertEqual(self.tracker.get_track(1).status, TrackStatus.TENTATIVE)  # type: ignore[union-attr]

        # Hit 3 -> CONFIRMED
        self.tracker.update([obs(0.2)], timestamp=0.2)
        self.assertEqual(self.tracker.get_track(1).status, TrackStatus.CONFIRMED)  # type: ignore[union-attr]

    def test_temporary_disappearance_and_recovery(self) -> None:
        """Confirmed track becomes LOST when missing observations, but recovers to CONFIRMED upon re-detection."""
        obs = lambda t, z: PhysicalObservation(
            position=Vector2(0.0, z),
            measurement_covariance=((0.04, 0.0), (0.0, 0.04)),
            confidence=0.9,
            timestamp=t,
            detection_id="d1",
            class_name="person",
        )

        # Confirm track over 3 frames
        for step in range(3):
            self.tracker.update([obs(step * 0.1, 4.0 - step * 0.1)], timestamp=step * 0.1)
        self.assertEqual(self.tracker.get_track(1).status, TrackStatus.CONFIRMED)  # type: ignore[union-attr]

        # Miss 2 frames -> Transitions to LOST (not deleted)
        self.tracker.update([], timestamp=0.3)
        self.assertEqual(self.tracker.get_track(1).status, TrackStatus.LOST)  # type: ignore[union-attr]
        self.tracker.update([], timestamp=0.4)
        self.assertEqual(self.tracker.get_track(1).status, TrackStatus.LOST)  # type: ignore[union-attr]

        # Re-detected at t=0.5 -> Recovers to CONFIRMED with same track_id = 1
        res = self.tracker.update([obs(0.5, 3.5)], timestamp=0.5)
        self.assertEqual(len(res.tracked_observations), 1)
        self.assertEqual(res.tracked_observations[0].track_id, 1)
        self.assertEqual(self.tracker.get_track(1).status, TrackStatus.CONFIRMED)  # type: ignore[union-attr]

    def test_stale_track_deletion(self) -> None:
        """Track exceeding max_missed_frames is deleted."""
        obs = PhysicalObservation(
            position=Vector2(0.0, 4.0),
            measurement_covariance=((0.04, 0.0), (0.0, 0.04)),
            confidence=0.9,
            timestamp=0.0,
            detection_id="d1",
            class_name="person",
        )
        # Confirm track
        for step in range(3):
            self.tracker.update([obs], timestamp=step * 0.1)

        # Miss 4 frames (max_missed_frames = 3)
        for step in range(3, 7):
            self.tracker.update([], timestamp=step * 0.1)

        self.assertEqual(self.tracker.get_track(1).status, TrackStatus.DELETED)  # type: ignore[union-attr]
        self.assertEqual(len(self.tracker.active_tracks), 0)


if __name__ == "__main__":
    unittest.main()
