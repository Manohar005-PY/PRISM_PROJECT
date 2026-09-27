"""State estimation evaluation engine for PRISM Part 17."""

from __future__ import annotations

import math
from typing import Any, Sequence

from simulation.evaluation.alignment import AlignedFrame
from simulation.evaluation.models import MetricStats


def compute_metric_stats(values: Sequence[float]) -> MetricStats:
    """Compute comprehensive descriptive statistics for a list of numerical values."""
    if not values:
        return MetricStats(n=0)

    clean_vals = [float(v) for v in values if math.isfinite(v)]
    n = len(clean_vals)
    if n == 0:
        return MetricStats(n=0)

    clean_vals.sort()
    mean_val = sum(clean_vals) / n
    variance = sum((x - mean_val) ** 2 for x in clean_vals) / n
    std_val = math.sqrt(variance)

    # Median and Percentiles
    def percentile(p: float) -> float:
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return clean_vals[int(k)]
        return clean_vals[int(f)] * (c - k) + clean_vals[int(c)] * (k - f)

    median_val = percentile(0.50)
    p25_val = percentile(0.25)
    p75_val = percentile(0.75)
    p95_val = percentile(0.95)

    mae_val = sum(abs(x) for x in clean_vals) / n
    rmse_val = math.sqrt(sum(x ** 2 for x in clean_vals) / n)

    return MetricStats(
        n=n,
        mean=round(mean_val, 4),
        median=round(median_val, 4),
        std=round(std_val, 4),
        min=round(clean_vals[0], 4),
        max=round(clean_vals[-1], 4),
        p25=round(p25_val, 4),
        p75=round(p75_val, 4),
        p95=round(p95_val, 4),
        mae=round(mae_val, 4),
        rmse=round(rmse_val, 4),
    )


class StateEstimationEvaluator:
    """Evaluates position and velocity errors between PRISM estimates and ground truth."""

    def evaluate_run(self, aligned_frames: Sequence[AlignedFrame]) -> dict[str, Any]:
        """Compute position and velocity error statistics across all matched objects in a run."""
        pos_errors_2d: list[float] = []
        pos_errors_x: list[float] = []
        pos_errors_z: list[float] = []

        vel_errors_2d: list[float] = []
        vel_errors_vx: list[float] = []
        vel_errors_vz: list[float] = []

        for frame in aligned_frames:
            for gt_obj, trk in frame.matched_pairs:
                # Position errors
                dx = trk["x"] - gt_obj["x"]
                dz = trk["z"] - gt_obj["z"]
                dist_err = math.sqrt(dx * dx + dz * dz)

                pos_errors_2d.append(dist_err)
                pos_errors_x.append(abs(dx))
                pos_errors_z.append(abs(dz))

                # Velocity errors
                dvx = trk["vx"] - gt_obj["vx"]
                dvz = trk["vz"] - gt_obj["vz"]
                vel_err = math.sqrt(dvx * dvx + dvz * dvz)

                vel_errors_2d.append(vel_err)
                vel_errors_vx.append(abs(dvx))
                vel_errors_vz.append(abs(dvz))

        return {
            "position_2d": compute_metric_stats(pos_errors_2d).to_dict(),
            "position_x": compute_metric_stats(pos_errors_x).to_dict(),
            "position_z": compute_metric_stats(pos_errors_z).to_dict(),
            "velocity_2d": compute_metric_stats(vel_errors_2d).to_dict(),
            "velocity_vx": compute_metric_stats(vel_errors_vx).to_dict(),
            "velocity_vz": compute_metric_stats(vel_errors_vz).to_dict(),
        }
