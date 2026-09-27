"""Unit tests for WorldState domain type."""

from dataclasses import FrozenInstanceError
import unittest

from app.core.types import ObjectState, Vector2, WorldState


class TestWorldState(unittest.TestCase):
    """Test suite for WorldState."""

    def setUp(self) -> None:
        self.state1 = ObjectState(
            object_id=1,
            class_id="person",
            position=Vector2(0.0, 5.0),
            velocity=Vector2(0.0, -1.0),
            timestamp=10.0,
        )
        self.state2 = ObjectState(
            object_id=2,
            class_id="chair",
            position=Vector2(1.5, 2.0),
            velocity=Vector2(0.0, 0.0),
            timestamp=10.0,
        )

    def test_valid_empty_world_state(self) -> None:
        """Test creating an empty world state."""
        world = WorldState(timestamp=10.0)
        self.assertEqual(world.timestamp, 10.0)
        self.assertEqual(len(world.objects), 0)

    def test_valid_world_state_with_objects(self) -> None:
        """Test creating world state with tracked objects."""
        world = WorldState(
            timestamp=10.0,
            objects={1: self.state1, 2: self.state2},
        )
        self.assertEqual(world.timestamp, 10.0)
        self.assertEqual(len(world.objects), 2)
        self.assertEqual(world.objects[1], self.state1)
        self.assertEqual(world.objects[2], self.state2)

    def test_objects_mapping_is_immutable(self) -> None:
        """Test that objects in WorldState cannot be mutated externally or via mapping."""
        obj_dict = {1: self.state1}
        world = WorldState(timestamp=10.0, objects=obj_dict)

        # Mutating original dict should not affect WorldState
        obj_dict[2] = self.state2
        self.assertEqual(len(world.objects), 1)

        # Direct mutation of mapping should raise TypeError
        with self.assertRaises(TypeError):
            world.objects[1] = self.state2  # type: ignore[index]

    def test_invalid_types_raise(self) -> None:
        """Test that invalid object keys or values raise TypeError."""
        with self.assertRaises(TypeError):
            WorldState(timestamp=1.0, objects={"not_an_int": self.state1})  # type: ignore[dict-item]

        with self.assertRaises(TypeError):
            WorldState(timestamp=1.0, objects={1: "not_a_state"})  # type: ignore[dict-item]

    def test_immutability(self) -> None:
        """Test that WorldState attributes cannot be modified."""
        world = WorldState(timestamp=10.0)
        with self.assertRaises(FrozenInstanceError):
            world.timestamp = 20.0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
