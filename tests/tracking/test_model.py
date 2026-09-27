"""Unit tests for tracking domain models (TrackRecord, AssociationPair, TrackingResult)."""

import unittest

from app.core.types import Vector2
from app.perception.model import PhysicalObservation, TrackedObservation
from app.tracking.model import (
    AssociationPair,
    TrackRecord,
    TrackStatus,
    TrackingResult,
)


class TestTrackingModel(unittest.TestCase):
    """Test suite for tracking domain data models."""

    def test_track_record_creation_and_immutability(self) -> None:
        track = TrackRecord(
            track_id=1,
            class_name="person",
            status=TrackStatus.TENTATIVE,
            first_timestamp=0.0,
            last_timestamp=0.0,
            age=1,
            hits=1,
            consecutive_hits=1,
            missed_frames=0,
        )
        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.class_name, "person")
        self.assertEqual(track.status, TrackStatus.TENTATIVE)
        self.assertEqual(track.first_timestamp, 0.0)
        self.assertEqual(track.created_at, 0.0)

        with self.assertRaises(Exception):
            track.hits = 2  # type: ignore[misc]

    def test_association_pair_creation(self) -> None:
        assoc = AssociationPair(
            track_id=1,
            observation_id="det_1",
            cost=2.35,
            distance=2.35,
            class_compatible=True,
        )
        self.assertEqual(assoc.track_id, 1)
        self.assertEqual(assoc.observation_id, "det_1")
        self.assertAlmostEqual(assoc.cost, 2.35)
        self.assertTrue(assoc.class_compatible)

    def test_tracking_result_fields(self) -> None:
        obs = PhysicalObservation(
            position=Vector2(0.0, 3.0),
            measurement_covariance=((0.04, 0.0), (0.0, 0.09)),
            confidence=0.9,
            timestamp=0.5,
            detection_id="det_1",
            class_name="person",
        )
        tracked_obs = TrackedObservation(track_id=1, observation=obs)
        track = TrackRecord(
            track_id=1,
            class_name="person",
            status=TrackStatus.CONFIRMED,
            first_timestamp=0.0,
            last_timestamp=0.5,
        )
        assoc = AssociationPair(track_id=1, observation_id="det_1", cost=0.1, distance=0.1)

        res = TrackingResult(
            confirmed_tracks=(track,),
            new_tracks=(),
            lost_tracks=(),
            deleted_tracks=(),
            associations=(assoc,),
            unmatched_observations=(),
            tracked_observations=(tracked_obs,),
            active_tracks=(track,),
            timestamp=0.5,
        )
        self.assertEqual(len(res.tracked_observations), 1)
        self.assertEqual(len(res.confirmed_tracks), 1)
        self.assertEqual(len(res.associations), 1)


if __name__ == "__main__":
    unittest.main()
