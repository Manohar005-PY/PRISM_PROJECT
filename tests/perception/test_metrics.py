"""Unit tests for perception and observation evaluation metrics."""

import math
import unittest

from app.core.types import Vector2
from app.perception.metrics import (
    contact_pixel_error,
    normalized_position_error,
    position_rmse,
    valid_projection_rate,
    x_rmse,
    z_rmse,
)


class TestPerceptionMetrics(unittest.TestCase):
    """Test suite for perception evaluation metrics."""

    def test_contact_pixel_error(self) -> None:
        err = contact_pixel_error((320.0, 240.0), (323.0, 244.0))
        self.assertAlmostEqual(err, 5.0)

    def test_valid_projection_rate(self) -> None:
        self.assertEqual(valid_projection_rate(8, 10), 80.0)
        self.assertEqual(valid_projection_rate(0, 0), 0.0)

    def test_rmse_metrics(self) -> None:
        meas = [Vector2(1.0, 3.0), Vector2(0.0, 4.0)]
        gt = [Vector2(1.2, 3.0), Vector2(0.0, 4.2)]

        # dx1=0.2, dz1=0.0 => d1^2 = 0.04
        # dx2=0.0, dz2=0.2 => d2^2 = 0.04
        # mean_sq = 0.04 => sqrt = 0.2
        self.assertAlmostEqual(position_rmse(meas, gt), 0.2)
        self.assertAlmostEqual(x_rmse(meas, gt), math.sqrt(0.04 / 2.0))
        self.assertAlmostEqual(z_rmse(meas, gt), math.sqrt(0.04 / 2.0))

    def test_normalized_position_error(self) -> None:
        meas = Vector2(1.2, 3.4)
        gt = Vector2(1.0, 3.0)
        cov = ((0.04, 0.0), (0.0, 0.16))

        # dx = 0.2 => dx^2 = 0.04 / 0.04 = 1.0
        # dz = 0.4 => dz^2 = 0.16 / 0.16 = 1.0
        # normalized error = 1.0 + 1.0 = 2.0
        norm_err = normalized_position_error(meas, gt, cov)
        self.assertAlmostEqual(norm_err, 2.0)


if __name__ == "__main__":
    unittest.main()
