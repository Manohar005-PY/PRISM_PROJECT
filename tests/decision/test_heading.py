"""Unit tests for safe heading geometry, clearance calculation, and best heading selection."""

import math
import unittest

from app.core.types import ObjectState, Vector2
from app.decision.context import DecisionContext
from app.decision.heading import SafeHeading, compute_safe_headings, select_best_heading


class TestSafeHeadingGeometry(unittest.TestCase):
    """Test suite for safe heading calculations."""

    def test_unobstructed_environment(self) -> None:
        """In open space, forward ray has maximum clearance and zero deviation."""
        ctx = DecisionContext(predicted_obstacles=())
        headings = compute_safe_headings(ctx, num_samples=11, max_lookahead=6.0)

        self.assertEqual(len(headings), 11)
        for h in headings:
            self.assertTrue(h.is_valid)
            self.assertAlmostEqual(h.clearance, 6.0)

        best = select_best_heading(headings, ctx)
        self.assertIsNotNone(best)
        self.assertAlmostEqual(best.angle, 0.0)  # type: ignore[union-attr]

    def test_obstacle_directly_ahead(self) -> None:
        """Obstacle at (0, 2) blocks the forward ray (angle=0), but left and right are safe."""
        obs = ObjectState(1, "person", Vector2(0.0, 2.0), Vector2(0.0, 0.0), 0.0)
        ctx = DecisionContext(predicted_obstacles=(obs,), minimum_clearance=1.5)
        headings = compute_safe_headings(ctx, num_samples=37, max_lookahead=6.0)

        # Forward heading (angle ~ 0) should be blocked (clearance ~ 2.0 - 0.7 = 1.3 < 1.5)
        forward = next(h for h in headings if abs(h.angle) <= 0.05)
        self.assertFalse(forward.is_valid)
        self.assertLess(forward.clearance, 1.5)

        # Large left/right angles must remain clear
        far_left = next(h for h in headings if h.angle <= -0.5)
        self.assertTrue(far_left.is_valid)

        best = select_best_heading(headings, ctx)
        self.assertIsNotNone(best)
        # Best heading should evade left or right
        self.assertGreater(abs(best.angle), 0.2)  # type: ignore[union-attr]

    def test_all_directions_blocked_returns_none(self) -> None:
        """When obstacles surround the user within minimum clearance, no safe heading exists."""
        obstacles = (
            ObjectState(1, "p", Vector2(0.0, 0.8), Vector2(0, 0), 0.0),    # Forward
            ObjectState(2, "p", Vector2(-0.8, 0.5), Vector2(0, 0), 0.0),   # Left
            ObjectState(3, "p", Vector2(0.8, 0.5), Vector2(0, 0), 0.0),    # Right
        )
        ctx = DecisionContext(predicted_obstacles=obstacles, minimum_clearance=1.5)
        headings = compute_safe_headings(ctx, num_samples=37, max_lookahead=6.0)

        best = select_best_heading(headings, ctx)
        self.assertIsNone(best)


if __name__ == "__main__":
    unittest.main()
