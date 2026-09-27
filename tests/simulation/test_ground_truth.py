"""Unit tests for GroundTruthFrame and GroundTruthSequence."""

from dataclasses import FrozenInstanceError
import unittest

from app.core.types import Vector2
from simulation.entities import EgoEntity, SimulatedEntity
from simulation.ground_truth import GroundTruthFrame, GroundTruthSequence


class TestGroundTruth(unittest.TestCase):
    """Test suite for ground-truth history tracking."""

    def setUp(self) -> None:
        self.ego0 = EgoEntity(position=Vector2(0.0, 0.0), velocity=Vector2(0.0, 1.0), timestamp=0.0)
        self.ego1 = EgoEntity(position=Vector2(0.0, 0.1), velocity=Vector2(0.0, 1.0), timestamp=0.1)
        self.obj1_0 = SimulatedEntity(object_id=1, class_id="person", position=Vector2(0.0, 5.0), velocity=Vector2(0.0, -1.0), timestamp=0.0)
        self.obj1_1 = SimulatedEntity(object_id=1, class_id="person", position=Vector2(0.0, 4.9), velocity=Vector2(0.0, -1.0), timestamp=0.1)
        self.obj2_0 = SimulatedEntity(object_id=2, class_id="chair", position=Vector2(2.0, 3.0), velocity=Vector2(0.0, 0.0), timestamp=0.0)
        self.obj2_1 = SimulatedEntity(object_id=2, class_id="chair", position=Vector2(2.0, 3.0), velocity=Vector2(0.0, 0.0), timestamp=0.1)

    def test_frame_construction_and_queries(self) -> None:
        """Test creating a GroundTruthFrame and querying objects by ID."""
        frame = GroundTruthFrame(
            timestamp=0.0,
            ego=self.ego0,
            objects=(self.obj1_0, self.obj2_0),
        )
        self.assertEqual(frame.timestamp, 0.0)
        self.assertEqual(frame.ego, self.ego0)
        self.assertEqual(len(frame.objects), 2)
        self.assertEqual(frame.get_object(1), self.obj1_0)
        self.assertEqual(frame.get_object(2), self.obj2_0)
        self.assertIsNone(frame.get_object(999))

    def test_sequence_monotonicity_validation(self) -> None:
        """Non-monotonic timestamps must raise ValueError."""
        f0 = GroundTruthFrame(timestamp=0.0, ego=self.ego0, objects=(self.obj1_0,))
        f1 = GroundTruthFrame(timestamp=0.1, ego=self.ego1, objects=(self.obj1_1,))
        f_bad = GroundTruthFrame(timestamp=0.05, ego=self.ego0, objects=(self.obj1_0,))

        # Valid sequence
        seq = GroundTruthSequence(frames=(f0, f1))
        self.assertEqual(seq.num_frames, 2)
        self.assertAlmostEqual(seq.duration, 0.1)

        # Invalid sequence (out of chronological order)
        with self.assertRaises(ValueError):
            GroundTruthSequence(frames=(f0, f1, f_bad))

        # Invalid sequence (duplicate timestamp)
        with self.assertRaises(ValueError):
            GroundTruthSequence(frames=(f0, f0))

    def test_sequence_trajectory_extraction(self) -> None:
        """Test extracting full history per object_id and ego across frames."""
        f0 = GroundTruthFrame(timestamp=0.0, ego=self.ego0, objects=(self.obj1_0, self.obj2_0))
        f1 = GroundTruthFrame(timestamp=0.1, ego=self.ego1, objects=(self.obj1_1, self.obj2_1))
        seq = GroundTruthSequence(frames=(f0, f1))

        obj1_traj = seq.get_object_trajectory(1)
        self.assertEqual(len(obj1_traj), 2)
        self.assertEqual(obj1_traj[0].position, Vector2(0.0, 5.0))
        self.assertEqual(obj1_traj[1].position, Vector2(0.0, 4.9))

        ego_traj = seq.get_ego_trajectory()
        self.assertEqual(len(ego_traj), 2)
        self.assertEqual(ego_traj[0].position, Vector2(0.0, 0.0))
        self.assertEqual(ego_traj[1].position, Vector2(0.0, 0.1))

    def test_immutability(self) -> None:
        """Test that frames and sequences cannot be mutated."""
        frame = GroundTruthFrame(timestamp=0.0, ego=self.ego0, objects=(self.obj1_0,))
        with self.assertRaises(FrozenInstanceError):
            frame.timestamp = 1.0  # type: ignore[misc]

        seq = GroundTruthSequence(frames=(frame,))
        with self.assertRaises(FrozenInstanceError):
            seq.frames = ()  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
