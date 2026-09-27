"""Perception calibration: Depth-bin empirical error analysis vs predicted measurement covariance."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from app.core.types import Vector2
from app.perception.model import PhysicalObservation


@dataclass(frozen=True)
class DepthBinMetrics:
    """Empirical calibration metrics for a specific depth range."""

    depth_range: str
    sample_count: int
    rmse_x: float
    rmse_z: float
    mae_x: float
    mae_z: float
    bias_x: float
    bias_z: float
    predicted_sigma_x: float
    predicted_sigma_z: float
    calibration_ratio_x: float
    calibration_ratio_z: float
    coverage_1sigma_z: float
    coverage_2sigma_z: float
    calibration_status: str


class PerceptionCalibrator:
    """Evaluates projection accuracy and calibrates empirical uncertainty R against observed errors."""

    DEPTH_BINS = [
        (0.0, 2.0, "0-2m"),
        (2.0, 4.0, "2-4m"),
        (4.0, 6.0, "4-6m"),
        (6.0, 8.0, "6-8m"),
        (8.0, 50.0, "8+m"),
    ]

    def __init__(self, calibration_tolerance: float = 0.25) -> None:
        self.calibration_tolerance = calibration_tolerance
        self._samples: list[tuple[Vector2, PhysicalObservation]] = []  # (gt_pos, obs)

    def record_sample(self, gt_pos: Vector2, obs: PhysicalObservation) -> None:
        """Record paired ground-truth position and projected observation."""
        self._samples.append((gt_pos, obs))

    def evaluate_calibration(self) -> list[DepthBinMetrics]:
        """Compute depth-binned RMSE, MAE, bias, coverage, and compare with predicted covariance."""
        bin_results: list[DepthBinMetrics] = []

        for z_min, z_max, label in self.DEPTH_BINS:
            bin_samples = [
                (gt, obs)
                for gt, obs in self._samples
                if z_min <= gt.z < z_max
            ]

            if not bin_samples:
                bin_results.append(
                    DepthBinMetrics(
                        depth_range=label,
                        sample_count=0,
                        rmse_x=0.0,
                        rmse_z=0.0,
                        mae_x=0.0,
                        mae_z=0.0,
                        bias_x=0.0,
                        bias_z=0.0,
                        predicted_sigma_x=0.0,
                        predicted_sigma_z=0.0,
                        calibration_ratio_x=1.0,
                        calibration_ratio_z=1.0,
                        coverage_1sigma_z=0.0,
                        coverage_2sigma_z=0.0,
                        calibration_status="NO_DATA",
                    )
                )
                continue

            errors_x = [obs.position.x - gt.x for gt, obs in bin_samples]
            errors_z = [obs.position.z - gt.z for gt, obs in bin_samples]

            n = len(bin_samples)
            rmse_x = math.sqrt(sum(e**2 for e in errors_x) / n)
            rmse_z = math.sqrt(sum(e**2 for e in errors_z) / n)
            mae_x = sum(abs(e) for e in errors_x) / n
            mae_z = sum(abs(e) for e in errors_z) / n
            bias_x = sum(errors_x) / n
            bias_z = sum(errors_z) / n

            # Average predicted standard deviations from measurement covariance R
            pred_sig_x = sum(math.sqrt(obs.measurement_covariance[0][0]) for _, obs in bin_samples) / n
            pred_sig_z = sum(math.sqrt(obs.measurement_covariance[1][1]) for _, obs in bin_samples) / n

            # Ratio of observed error to predicted uncertainty
            cal_ratio_x = rmse_x / max(1e-4, pred_sig_x)
            cal_ratio_z = rmse_z / max(1e-4, pred_sig_z)

            # Empirical coverage calculation for depth Z
            in_1sig = 0
            in_2sig = 0
            for (gt, obs), ez in zip(bin_samples, errors_z):
                sig_z = math.sqrt(obs.measurement_covariance[1][1])
                if abs(ez) <= sig_z:
                    in_1sig += 1
                if abs(ez) <= 2.0 * sig_z:
                    in_2sig += 1

            cov_1sig = (in_1sig / n) * 100.0
            cov_2sig = (in_2sig / n) * 100.0

            # Scientific terminology:
            # ratio < 1 - tol => CONSERVATIVE (predicted uncertainty is larger than observed error)
            # ratio ~= 1 => CALIBRATED
            # ratio > 1 + tol => UNDER_ESTIMATING (predicted uncertainty is smaller than observed error)
            if cal_ratio_z > (1.0 + self.calibration_tolerance):
                status = "UNDER_ESTIMATING"
            elif cal_ratio_z < (1.0 - self.calibration_tolerance):
                status = "CONSERVATIVE"
            else:
                status = "CALIBRATED"

            bin_results.append(
                DepthBinMetrics(
                    depth_range=label,
                    sample_count=n,
                    rmse_x=rmse_x,
                    rmse_z=rmse_z,
                    mae_x=mae_x,
                    mae_z=mae_z,
                    bias_x=bias_x,
                    bias_z=bias_z,
                    predicted_sigma_x=pred_sig_x,
                    predicted_sigma_z=pred_sig_z,
                    calibration_ratio_x=cal_ratio_x,
                    calibration_ratio_z=cal_ratio_z,
                    coverage_1sigma_z=cov_1sig,
                    coverage_2sigma_z=cov_2sig,
                    calibration_status=status,
                )
            )

        return bin_results
