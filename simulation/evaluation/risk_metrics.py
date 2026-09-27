"""Predictive risk assessment and calibration evaluation engine for PRISM Part 17."""

from __future__ import annotations

import math
from typing import Any, Sequence

from simulation.evaluation.alignment import AlignedFrame
from simulation.evaluation.models import ClassificationStats, RiskCalibrationBin


RISK_CATEGORIES = ("SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL")


class RiskEvaluator:
    """Evaluates predictive risk scoring, multi-horizon event detection, and risk-to-event calibration."""

    def __init__(
        self,
        event_horizons: tuple[float, ...] = (1.0, 2.0, 3.0),
        risk_bins: tuple[tuple[float, float], ...] = (
            (0.0, 20.0),
            (20.0, 40.0),
            (40.0, 60.0),
            (60.0, 80.0),
            (80.0, 100.0),
        ),
    ) -> None:
        self.event_horizons: tuple[float, ...] = event_horizons
        self.risk_bins: tuple[tuple[float, float], ...] = risk_bins

    def evaluate_run(self, aligned_frames: Sequence[AlignedFrame]) -> dict[str, Any]:
        """Compute multi-horizon risk detection stats, confusion matrices, and calibration bin frequencies."""
        if not aligned_frames:
            return {
                "multi_horizon_detection": {},
                "risk_calibration_bins": [],
                "confusion_matrix": {},
            }

        num_frames = len(aligned_frames)
        dt = aligned_frames[1].timestamp - aligned_frames[0].timestamp if num_frames > 1 else 0.05

        # 1. Multi-Horizon Binary Collision Event Detection (H = 1s, 2s, 3s)
        horizon_stats: dict[str, Any] = {}

        for h in self.event_horizons:
            steps_ahead = int(round(h / dt))
            tp = tn = fp = fn = 0

            for f_idx, frame in enumerate(aligned_frames):
                # PRISM predicted danger: HIGH or CRITICAL risk (or risk_score >= 70)
                pr_risk_lvl = frame.prism_output.get("risk_level", "SAFE")
                pr_risk_score = frame.prism_output.get("risk_score", 0.0)
                pr_alert = (pr_risk_lvl in ("HIGH", "CRITICAL") or pr_risk_score >= 70.0)

                # Ground-truth event: does any true collision occur in [f_idx, f_idx + steps_ahead]?
                future_collision = False
                max_step = min(num_frames, f_idx + steps_ahead + 1)
                for f_future in range(f_idx, max_step):
                    if aligned_frames[f_future].gt_collision_state.get("has_collided", False):
                        future_collision = True
                        break

                if pr_alert and future_collision:
                    tp += 1
                elif not pr_alert and not future_collision:
                    tn += 1
                elif pr_alert and not future_collision:
                    fp += 1
                elif not pr_alert and future_collision:
                    fn += 1

            prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 and fp == 0 else 0.0)
            rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tp == 0 and fn == 0 else 0.0)
            f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            acc = (tp + tn) / max(1, tp + tn + fp + fn)

            horizon_stats[f"horizon_{h:.1f}s"] = ClassificationStats(
                tp=tp,
                tn=tn,
                fp=fp,
                fn=fn,
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1=round(f1, 4),
                accuracy=round(acc, 4),
            ).to_dict()

        # 2. Risk-to-Event Calibration Binning
        # Measures: If PRISM outputs score in bin B, how frequently does an actual collision occur within 3.0s?
        eval_h = 3.0
        steps_eval = int(round(eval_h / dt))
        bin_records: list[RiskCalibrationBin] = []

        for b_min, b_max in self.risk_bins:
            b_label = f"[{int(b_min)}-{int(b_max)}]"
            sample_count = 0
            event_count = 0

            for f_idx, frame in enumerate(aligned_frames):
                score = frame.prism_output.get("risk_score", 0.0)
                # Upper bound inclusive for top bin
                in_bin = (b_min <= score <= b_max) if b_max == 100.0 else (b_min <= score < b_max)

                if in_bin:
                    sample_count += 1
                    max_step = min(num_frames, f_idx + steps_eval + 1)
                    if any(aligned_frames[k].gt_collision_state.get("has_collided", False) for k in range(f_idx, max_step)):
                        event_count += 1

            freq = round(event_count / sample_count, 4) if sample_count > 0 else None
            bin_records.append(
                RiskCalibrationBin(
                    bin_label=b_label,
                    min_score=b_min,
                    max_score=b_max,
                    sample_count=sample_count,
                    collision_event_count=event_count,
                    empirical_collision_frequency=freq,
                )
            )

        # 3. Categorical Distribution Breakdown
        risk_level_counts: dict[str, int] = {cat: 0 for cat in RISK_CATEGORIES}
        for frame in aligned_frames:
            lvl = frame.prism_output.get("risk_level", "SAFE")
            if lvl in risk_level_counts:
                risk_level_counts[lvl] += 1
            else:
                risk_level_counts[lvl] = 1

        return {
            "multi_horizon_detection": horizon_stats,
            "risk_calibration_bins": [b.to_dict() for b in bin_records],
            "risk_level_distribution": risk_level_counts,
        }
