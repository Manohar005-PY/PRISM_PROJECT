"""Deterministic Gaussian noise injection for synthetic observations.

Implements controlled measurement noise using standard library random.Random.
Explicit random seeding guarantees reproducible simulation runs for the same
configuration and Python/runtime environment.
"""

from __future__ import annotations

import math
import random

from app.core.types import Vector2


class GaussianNoiseModel:
    """Zero-mean Gaussian noise model for ground-plane position observations.

    Attributes:
        sigma_x: Standard deviation of lateral position measurement noise (meters).
        sigma_z: Standard deviation of forward/depth position measurement noise (meters).
        seed: Seed for the pseudo-random generator ensuring determinism.
    """

    def __init__(
        self,
        sigma_x: float = 0.0,
        sigma_z: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if not isinstance(sigma_x, (int, float)) or not math.isfinite(sigma_x) or sigma_x < 0.0:
            raise ValueError(f"sigma_x must be a non-negative finite float, got {sigma_x!r}")
        if not isinstance(sigma_z, (int, float)) or not math.isfinite(sigma_z) or sigma_z < 0.0:
            raise ValueError(f"sigma_z must be a non-negative finite float, got {sigma_z!r}")
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"seed must be an int or None, got {type(seed)}")

        self.sigma_x: float = float(sigma_x)
        self.sigma_z: float = float(sigma_z)
        self.seed: int | None = seed
        self._rng: random.Random = random.Random(seed)

    def reset_seed(self, seed: int | None = None) -> None:
        """Reset the internal RNG to a specific seed or reseed with initial seed."""
        if seed is not None:
            self.seed = seed
        self._rng.seed(self.seed)

    def apply_position_noise(self, position: Vector2) -> Vector2:
        """Apply zero-mean Gaussian noise to a true ground-plane position.

        Args:
            position: Exact ground-truth position Vector2.

        Returns:
            Noisy position Vector2.
        """
        if not isinstance(position, Vector2):
            raise TypeError(f"position must be a Vector2, got {type(position)}")

        noise_x = self._rng.gauss(0.0, self.sigma_x) if self.sigma_x > 0.0 else 0.0
        noise_z = self._rng.gauss(0.0, self.sigma_z) if self.sigma_z > 0.0 else 0.0

        return Vector2(
            x=position.x + noise_x,
            z=position.z + noise_z,
        )
