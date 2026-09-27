"""Unit tests for GaussianNoiseModel."""

import math
import unittest

from app.core.types import Vector2
from simulation.noise import GaussianNoiseModel


class TestGaussianNoiseModel(unittest.TestCase):
    """Test suite for seeded Gaussian measurement noise."""

    def test_zero_noise_preserves_ground_truth(self) -> None:
        """When sigma is zero, position is completely unchanged."""
        model = GaussianNoiseModel(sigma_x=0.0, sigma_z=0.0, seed=42)
        pos = Vector2(1.23, 4.56)
        noisy = model.apply_position_noise(pos)
        self.assertEqual(noisy, pos)

    def test_seed_reproducibility(self) -> None:
        """Two noise models with identical seed produce the exact same sequence."""
        model1 = GaussianNoiseModel(sigma_x=0.1, sigma_z=0.2, seed=12345)
        model2 = GaussianNoiseModel(sigma_x=0.1, sigma_z=0.2, seed=12345)

        pos = Vector2(0.0, 5.0)
        for _ in range(20):
            res1 = model1.apply_position_noise(pos)
            res2 = model2.apply_position_noise(pos)
            self.assertEqual(res1, res2)

    def test_different_seeds_produce_different_noise(self) -> None:
        """Different seeds must produce different sequences."""
        model1 = GaussianNoiseModel(sigma_x=0.5, sigma_z=0.5, seed=10)
        model2 = GaussianNoiseModel(sigma_x=0.5, sigma_z=0.5, seed=20)

        pos = Vector2(0.0, 5.0)
        outputs1 = [model1.apply_position_noise(pos) for _ in range(5)]
        outputs2 = [model2.apply_position_noise(pos) for _ in range(5)]
        self.assertNotEqual(outputs1, outputs2)

    def test_reset_seed(self) -> None:
        """Resetting the seed restores the initial pseudo-random sequence."""
        model = GaussianNoiseModel(sigma_x=0.2, sigma_z=0.2, seed=999)
        pos = Vector2(1.0, 2.0)

        seq1 = [model.apply_position_noise(pos) for _ in range(10)]
        model.reset_seed()
        seq2 = [model.apply_position_noise(pos) for _ in range(10)]
        self.assertEqual(seq1, seq2)

    def test_statistical_properties(self) -> None:
        """Verify sample mean and std dev approximate configured zero-mean Gaussian over a reasonable sample size."""
        sigma_x = 0.3
        sigma_z = 0.5
        model = GaussianNoiseModel(sigma_x=sigma_x, sigma_z=sigma_z, seed=42)
        origin = Vector2(0.0, 0.0)

        n_samples = 2000
        samples = [model.apply_position_noise(origin) for _ in range(n_samples)]

        mean_x = sum(s.x for s in samples) / n_samples
        mean_z = sum(s.z for s in samples) / n_samples

        var_x = sum((s.x - mean_x) ** 2 for s in samples) / (n_samples - 1)
        var_z = sum((s.z - mean_z) ** 2 for s in samples) / (n_samples - 1)
        std_x = math.sqrt(var_x)
        std_z = math.sqrt(var_z)

        # Tolerance checks
        self.assertAlmostEqual(mean_x, 0.0, delta=0.05)
        self.assertAlmostEqual(mean_z, 0.0, delta=0.05)
        self.assertAlmostEqual(std_x, sigma_x, delta=0.05)
        self.assertAlmostEqual(std_z, sigma_z, delta=0.05)

    def test_invalid_parameters_raise(self) -> None:
        """Negative sigma or invalid types must raise errors."""
        with self.assertRaises(ValueError):
            GaussianNoiseModel(sigma_x=-0.1, sigma_z=0.1)

        with self.assertRaises(ValueError):
            GaussianNoiseModel(sigma_x=0.1, sigma_z=-0.1)

        with self.assertRaises(ValueError):
            GaussianNoiseModel(sigma_x=float("nan"), sigma_z=0.1)

        with self.assertRaises(TypeError):
            GaussianNoiseModel(sigma_x=0.1, sigma_z=0.1, seed="invalid")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
