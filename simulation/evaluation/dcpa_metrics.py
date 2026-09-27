"""Closest Point of Approach (dCPA) evaluation engine for PRISM Part 17."""

from __future__ import annotations

import math
from typing import Any, Sequence

from simulation.evaluation.alignment import AlignedFrame
from simulation.evaluation.models import MetricStats
from simulation.evaluation.state_metrics import compute_metric_stats


class DCPAEvaluator:
    """Evaluates PRISM estimated dCPA against the independent ground-truth analytical dCPA."""

    def evaluate_run(self, aligned_frames: Sequence[AlignedFrame]) -> dict[str, Any]:
        """Compute dCPA estimation error statistics and bias."""
        abs_errors: list[float] = []
        signed_errors: list[float] = []

        for frame in aligned_frames:
            gt_col = frame.gt_collision_state
            gt_dcpa = gt_col.get("closest_point_of_approach")
            pr_dcpa = frame.prism_output.get("predicted_cpa")

            if gt_dcpa is not None and pr_dcpa is not None:
                if math.isfinite(gt_dcpa) and math.isfinite(pr_dcpa):
                    err = pr_dcpa - gt_dcpa
                    abs_errors.append(abs(err))
                    signed_errors.append(err)

        stats = compute_metric_stats(abs_errors)
        bias = (sum(signed_errors) / len(signed_errors)) if signed_errors else None

        return {
            "error_stats": stats.to_dict(),
            "bias_m": round(bias, 4) if bias is not None else None,
        }
