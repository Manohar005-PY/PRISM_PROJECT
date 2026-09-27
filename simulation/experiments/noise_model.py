"""Synthetic observation and measurement noise generator for PRISM Part 16.

CRITICAL ARCHITECTURAL ROLE:
This module acts as the synthetic perception boundary adapter for controlled experiments.
It replaces neural object detectors (e.g. YOLO) by generating realistic PhysicalObservation
instances with configurable Gaussian measurement noise and observation dropout probabilities.
This isolates the downstream PRISM predictive pipeline (tracking, Kalman filter, trajectory
prediction, TTC/dCPA, risk assessment, avoidance policy) from upstream detector variability.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from app.core.types import Vector2
from app.perception.model import PhysicalObservation
from simulation.experiments.models import (
    GroundTruthObjectState,
    NoiseConfiguration,
    ObservationItem,
)


class SyntheticPerceptionModel:
    """Generates synthetic PhysicalObservation instances from ground-truth states."""

    def __init__(self, config: NoiseConfiguration) -> None:
        self.config: NoiseConfiguration = config
        self._rng: random.Random = random.Random(config.seed)

    def reseed(self, seed: int | None = None) -> None:
        """Reset internal pseudo-random number generator for determinism."""
        if seed is not None:
            self._rng.seed(seed)
        else:
            self._rng.seed(self.config.seed)

    def generate_observations(
        self,
        ground_truth_objects: Sequence[GroundTruthObjectState],
        timestamp: float,
    ) -> tuple[list[ObservationItem], list[PhysicalObservation]]:
        """Generate noisy observations and valid PhysicalObservation instances for PRISM.

        Args:
            ground_truth_objects: Exact physical ground-truth states at time t.
            timestamp: Monotonic simulation time in seconds.

        Returns:
            tuple (observation_items, physical_observations)
        """
        obs_items: list[ObservationItem] = []
        prism_obs: list[PhysicalObservation] = []

        # Construct 2x2 measurement covariance R
        var_x = max(0.005, self.config.sigma_position_x ** 2) if self.config.enabled else 0.01
        var_z = max(0.005, self.config.sigma_position_z ** 2) if self.config.enabled else 0.01
        cov: tuple[tuple[float, float], tuple[float, float]] = ((var_x, 0.0), (0.0, var_z))

        for obj in ground_truth_objects:
            # Check for observation dropout
            if self.config.enabled and self.config.dropout_probability > 0.0:
                draw = self._rng.random()
                if draw < self.config.dropout_probability:
                    # Observation dropped in this frame
                    obs_items.append(
                        ObservationItem(
                            object_id=obj.object_id,
                            class_id=obj.class_id,
                            x=obj.x,
                            z=obj.z,
                            covariance=cov,
                            confidence=0.0,
                            valid=False,
                        )
                    )
                    continue

            # Injected Gaussian noise
            if self.config.enabled and (self.config.sigma_position_x > 0.0 or self.config.sigma_position_z > 0.0):
                noise_x = self._rng.gauss(0.0, self.config.sigma_position_x) if self.config.sigma_position_x > 0.0 else 0.0
                noise_z = self._rng.gauss(0.0, self.config.sigma_position_z) if self.config.sigma_position_z > 0.0 else 0.0
            else:
                noise_x = 0.0
                noise_z = 0.0

            obs_x = round(obj.x + noise_x, 4)
            obs_z = round(obj.z + noise_z, 4)
            confidence = 0.95

            item = ObservationItem(
                object_id=obj.object_id,
                class_id=obj.class_id,
                x=obs_x,
                z=obs_z,
                covariance=cov,
                confidence=confidence,
                valid=True,
            )
            obs_items.append(item)

            # Convert to PRISM domain PhysicalObservation
            p_obs = PhysicalObservation(
                position=Vector2(x=obs_x, z=obs_z),
                measurement_covariance=cov,
                confidence=confidence,
                timestamp=timestamp,
                detection_id=f"det_{obj.object_id}_{int(round(timestamp * 1000))}",
                class_name=str(obj.class_id),
            )
            prism_obs.append(p_obs)

        return obs_items, prism_obs
