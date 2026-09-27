"""Observation uncertainty modeling: distance-dependent depth scaling and confidence weighting."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MeasurementUncertaintyConfig:
    """Parameters for physical observation noise covariance calculation.

    Attributes:
        sigma_x0: Base lateral measurement standard deviation at near distance (meters).
        sigma_z0: Base longitudinal measurement standard deviation at near distance (meters).
        beta_x: Quadratic depth growth factor for lateral variance R_xx.
        beta_z: Quadratic depth growth factor for longitudinal variance R_zz.
        confidence_noise_gain: Empirical multiplier gamma for low-confidence variance scaling.
        min_confidence: Minimum accepted confidence threshold for valid detection.
    """

    sigma_x0: float = 0.15
    sigma_z0: float = 0.25
    beta_x: float = 0.02
    beta_z: float = 0.05
    confidence_noise_gain: float = 1.5
    min_confidence: float = 0.20

    def __post_init__(self) -> None:
        for name, val in [
            ("sigma_x0", self.sigma_x0),
            ("sigma_z0", self.sigma_z0),
            ("beta_x", self.beta_x),
            ("beta_z", self.beta_z),
            ("confidence_noise_gain", self.confidence_noise_gain),
            ("min_confidence", self.min_confidence),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val!r}")

        if self.sigma_x0 <= 0.0:
            raise ValueError(f"sigma_x0 must be strictly positive, got {self.sigma_x0}")
        if self.sigma_z0 <= 0.0:
            raise ValueError(f"sigma_z0 must be strictly positive, got {self.sigma_z0}")
        if self.beta_x < 0.0:
            raise ValueError(f"beta_x must be non-negative, got {self.beta_x}")
        if self.beta_z < 0.0:
            raise ValueError(f"beta_z must be non-negative, got {self.beta_z}")
        if self.confidence_noise_gain < 0.0:
            raise ValueError(f"confidence_noise_gain must be non-negative, got {self.confidence_noise_gain}")
        if not (0.0 < self.min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be in (0.0, 1.0], got {self.min_confidence}")


class MeasurementUncertaintyModel:
    """Calculates 2x2 measurement covariance matrix R for ground-plane observations.

    Formulation:
        R_xx(Z, c) = sigma_x0^2 * (1 + beta_x * Z^2) * s(c)^2
        R_zz(Z, c) = sigma_z0^2 * (1 + beta_z * Z^2) * s(c)^2
        where s(c) = 1.0 + gamma * (1.0 - c)

    Note:
        Confidence scaling is an empirical configurable heuristic, not a calibrated probabilistic mapping.
    """

    def __init__(self, config: MeasurementUncertaintyConfig | None = None) -> None:
        self.config = config if config is not None else MeasurementUncertaintyConfig()

    def confidence_scale_factor(self, confidence: float) -> float:
        """Compute bounded, monotonic uncertainty scaling factor based on detector confidence.

        Formula: s(c) = 1.0 + gamma * (1.0 - c)
        Range: [1.0, 1.0 + gamma] for c in [0.0, 1.0].
        """
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
            raise ValueError(f"confidence must be a finite float, got {confidence!r}")
        clamped_c = max(0.0, min(1.0, float(confidence)))
        return 1.0 + self.config.confidence_noise_gain * (1.0 - clamped_c)

    def compute_covariance(
        self,
        depth_z: float,
        confidence: float = 1.0,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Calculate 2x2 measurement noise covariance matrix R = diag(R_xx, R_zz).

        Args:
            depth_z: Measured forward physical depth in meters (must be > 0).
            confidence: Detector confidence score in [0.0, 1.0].

        Returns:
            2x2 covariance tuple ((R_xx, 0.0), (0.0, R_zz)).

        Raises:
            ValueError: If depth_z <= 0 or non-finite.
        """
        if not isinstance(depth_z, (int, float)) or not math.isfinite(depth_z) or depth_z <= 0.0:
            raise ValueError(f"depth_z must be a positive finite float, got {depth_z!r}")

        scale = self.confidence_scale_factor(confidence)
        scale_sq = scale * scale

        # Consistent variance model with Part 4:
        # Var_base(Z) = sigma_0^2 * (1 + beta * Z^2)
        var_x = (self.config.sigma_x0 * self.config.sigma_x0) * (1.0 + self.config.beta_x * (depth_z * depth_z)) * scale_sq
        var_z = (self.config.sigma_z0 * self.config.sigma_z0) * (1.0 + self.config.beta_z * (depth_z * depth_z)) * scale_sq

        return ((var_x, 0.0), (0.0, var_z))
