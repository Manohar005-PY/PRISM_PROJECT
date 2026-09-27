"""Measurement noise covariance models for PRISM state estimation.

Provides configurable observation covariance matrix R representing measurement uncertainty
over ground-plane coordinates [X, Z].
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

from app.core.types import Vector2

Covariance2x2 = tuple[
    tuple[float, float],
    tuple[float, float],
]


@runtime_checkable
class MeasurementNoiseModel(Protocol):
    """Protocol for measurement noise covariance providers."""

    def measurement_covariance(self, measurement: Vector2) -> Covariance2x2:
        """Compute the 2x2 measurement noise covariance matrix R for a given measurement.

        Args:
            measurement: Measured ground-plane position Vector2(x, z).

        Returns:
            2x2 covariance tuple representing uncertainty over [X, Z].
        """
        ...


@dataclass(frozen=True)
class ConstantMeasurementNoise:
    """Constant measurement noise covariance model.

    R = [ [sigma_x^2, 0],
          [0, sigma_z^2] ]

    Attributes:
        sigma_x: Standard deviation of lateral position measurement (meters).
        sigma_z: Standard deviation of forward/depth position measurement (meters).
    """

    sigma_x: float = 0.1
    sigma_z: float = 0.2

    def __post_init__(self) -> None:
        if not isinstance(self.sigma_x, (int, float)) or not math.isfinite(self.sigma_x) or self.sigma_x <= 0.0:
            raise ValueError(f"sigma_x must be a positive finite float, got {self.sigma_x!r}")
        if not isinstance(self.sigma_z, (int, float)) or not math.isfinite(self.sigma_z) or self.sigma_z <= 0.0:
            raise ValueError(f"sigma_z must be a positive finite float, got {self.sigma_z!r}")
        object.__setattr__(self, "sigma_x", float(self.sigma_x))
        object.__setattr__(self, "sigma_z", float(self.sigma_z))

    def measurement_covariance(self, measurement: Vector2) -> Covariance2x2:
        """Return the constant 2x2 diagonal covariance matrix."""
        var_x = self.sigma_x * self.sigma_x
        var_z = self.sigma_z * self.sigma_z
        return (
            (var_x, 0.0),
            (0.0, var_z),
        )


@dataclass(frozen=True)
class DistanceDependentMeasurementNoise:
    """Distance-dependent measurement noise covariance model.

    Models the non-linear perspective growth of depth uncertainty at larger distances:
    sigma_z^2(Z) = sigma_z0^2 * (1 + alpha * Z^2)
    sigma_x^2(Z) = sigma_x0^2 * (1 + beta * Z^2)

    Attributes:
        sigma_x0: Baseline lateral measurement standard deviation at Z=0 (meters).
        sigma_z0: Baseline depth measurement standard deviation at Z=0 (meters).
        alpha: Quadratic depth uncertainty growth coefficient.
        beta: Quadratic lateral uncertainty growth coefficient.
    """

    sigma_x0: float = 0.1
    sigma_z0: float = 0.2
    alpha: float = 0.05
    beta: float = 0.01

    def __post_init__(self) -> None:
        for name, val in [("sigma_x0", self.sigma_x0), ("sigma_z0", self.sigma_z0)]:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be a positive finite float, got {val!r}")
        for name, val in [("alpha", self.alpha), ("beta", self.beta)]:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be a non-negative finite float, got {val!r}")
        object.__setattr__(self, "sigma_x0", float(self.sigma_x0))
        object.__setattr__(self, "sigma_z0", float(self.sigma_z0))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "beta", float(self.beta))

    def measurement_covariance(self, measurement: Vector2) -> Covariance2x2:
        """Compute the distance-scaled 2x2 covariance matrix based on forward depth Z."""
        if not isinstance(measurement, Vector2):
            raise TypeError(f"measurement must be a Vector2, got {type(measurement)}")

        z_clamped = max(0.0, measurement.z)
        z_sq = z_clamped * z_clamped

        var_x = (self.sigma_x0 * self.sigma_x0) * (1.0 + self.beta * z_sq)
        var_z = (self.sigma_z0 * self.sigma_z0) * (1.0 + self.alpha * z_sq)

        return (
            (var_x, 0.0),
            (0.0, var_z),
        )
