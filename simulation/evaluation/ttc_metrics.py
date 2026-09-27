"""Time-to-Collision (TTC) evaluation engine for PRISM Part 17."""

from __future__ import annotations

import math
from typing import Any, Sequence

from simulation.evaluation.alignment import AlignedFrame
from simulation.evaluation.models import MetricStats
from simulation.evaluation.state_metrics import compute_metric_stats


class TTCEvaluator:
    """Evaluates PRISM estimated TTC against the independent ground-truth analytical TTC."""

    def evaluate_run(self, aligned_frames: Sequence[AlignedFrame]) -> dict[str, Any]:
        """Compute TTC estimation error and finite/infinite disagreement classifications."""
        abs_errors: list[float] = []
        signed_errors: list[float] = []

        count_finite_vs_finite = 0
        count_false_predicted_collision = 0  # PRISM finite, GT inf/None
        count_missed_predicted_collision = 0  # PRISM inf/None, GT finite
        count_concordant_safe = 0  # Both inf/None

        tolerance_025s = 0
        tolerance_050s = 0
        tolerance_100s = 0

        for frame in aligned_frames:
            gt_col = frame.gt_collision_state
            gt_ttc = gt_col.get("time_to_collision")
            pr_ttc = frame.prism_output.get("predicted_ttc")

            gt_is_finite = (gt_ttc is not None and math.isfinite(gt_ttc))
            pr_is_finite = (pr_ttc is not None and math.isfinite(pr_ttc))

            if gt_is_finite and pr_is_finite:
                count_finite_vs_finite += 1
                err = pr_ttc - gt_ttc
                abs_err = abs(err)

                abs_errors.append(abs_err)
                signed_errors.append(err)

                if abs_err <= 0.25:
                    tolerance_025s += 1
                if abs_err <= 0.50:
                    tolerance_050s += 1
                if abs_err <= 1.00:
                    tolerance_100s += 1

            elif pr_is_finite and not gt_is_finite:
                count_false_predicted_collision += 1
            elif not pr_is_finite and gt_is_finite:
                count_missed_predicted_collision += 1
            else:
                count_concordant_safe += 1

        stats = compute_metric_stats(abs_errors)
        bias = (sum(signed_errors) / len(signed_errors)) if signed_errors else None

        n_finite = max(1, count_finite_vs_finite)
        ratio_025 = round(tolerance_025s / n_finite, 4) if count_finite_vs_finite > 0 else None
        ratio_050 = round(tolerance_050s / n_finite, 4) if count_finite_vs_finite > 0 else None
        ratio_100 = round(tolerance_100s / n_finite, 4) if count_finite_vs_finite > 0 else None

        return {
            "error_stats": stats.to_dict(),
            "bias_s": round(bias, 4) if bias is not None else None,
            "finite_vs_finite_count": count_finite_vs_finite,
            "false_predicted_collision_count": count_false_predicted_collision,
            "missed_predicted_collision_count": count_missed_predicted_collision,
            "concordant_safe_count": count_concordant_safe,
            "ratio_within_0_25s": ratio_025,
            "ratio_within_0_50s": ratio_050,
            "ratio_within_1_00s": ratio_100,
        }
