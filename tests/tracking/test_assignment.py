"""Unit tests for pure Python Hungarian algorithm (assignment.py)."""

import unittest

from app.tracking.assignment import assign_tracks_to_observations, linear_sum_assignment


class TestTrackingAssignment(unittest.TestCase):
    """Test suite for Hungarian global assignment algorithm."""

    def test_square_matrix_optimal_assignment(self) -> None:
        cost = [
            [4.0, 1.0, 3.0],
            [2.0, 0.5, 5.0],
            [3.0, 2.0, 2.0],
        ]
        # Optimal assignment:
        # Row 0 -> Col 0 (4) or Col 2 (3)?
        # If Row 1 -> Col 1 (0.5), Row 2 -> Col 2 (2.0), Row 0 -> Col 0 (4.0) => sum = 6.5
        # If Row 0 -> Col 1 (1.0), Row 1 -> Col 0 (2.0), Row 2 -> Col 2 (2.0) => sum = 5.0 (Optimal!)
        matches = linear_sum_assignment(cost)
        match_dict = dict(matches)

        self.assertEqual(match_dict[0], 1)
        self.assertEqual(match_dict[1], 0)
        self.assertEqual(match_dict[2], 2)

    def test_rectangular_matrices(self) -> None:
        # More tracks than observations (3 x 2)
        cost_3x2 = [
            [1.0, 10.0],
            [10.0, 2.0],
            [10.0, 10.0],
        ]
        matches, unassigned_trks, unassigned_obs = assign_tracks_to_observations(cost_3x2, max_cost_threshold=5.0)
        self.assertEqual(len(matches), 2)
        self.assertEqual(dict(matches)[0], 0)
        self.assertEqual(dict(matches)[1], 1)
        self.assertEqual(unassigned_trks, [2])
        self.assertEqual(unassigned_obs, [])

        # More observations than tracks (2 x 3)
        cost_2x3 = [
            [1.0, 10.0, 10.0],
            [10.0, 2.0, 10.0],
        ]
        matches, unassigned_trks, unassigned_obs = assign_tracks_to_observations(cost_2x3, max_cost_threshold=5.0)
        self.assertEqual(len(matches), 2)
        self.assertEqual(unassigned_trks, [])
        self.assertEqual(unassigned_obs, [2])

    def test_empty_matrices(self) -> None:
        matches, un_trks, un_obs = assign_tracks_to_observations([], max_cost_threshold=5.0)
        self.assertEqual(matches, [])
        self.assertEqual(un_trks, [])
        self.assertEqual(un_obs, [])


if __name__ == "__main__":
    unittest.main()
