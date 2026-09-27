"""Noise robustness and graceful degradation evaluation engine for PRISM Part 17."""

from __future__ import annotations

from typing import Any, Sequence

from simulation.evaluation.models import RunEvaluationResult


class RobustnessEvaluator:
    """Aggregates and compares performance metrics across controlled sensor noise conditions."""

    def evaluate_noise_conditions(
        self,
        run_results: Sequence[RunEvaluationResult],
    ) -> dict[str, dict[str, Any]]:
        """Compute performance summaries grouped by noise condition."""
        grouped: dict[str, list[RunEvaluationResult]] = {}

        for res in run_results:
            cond = res.noise_condition
            if cond not in grouped:
                grouped[cond] = []
            grouped[cond].append(res)

        noise_summaries: dict[str, dict[str, Any]] = {}

        for cond, runs in grouped.items():
            n_runs = len(runs)

            pos_maes = [r.state_estimation.get("position_2d", {}).get("mae") for r in runs if r.state_estimation.get("position_2d", {}).get("mae") is not None]
            vel_maes = [r.state_estimation.get("velocity_2d", {}).get("mae") for r in runs if r.state_estimation.get("velocity_2d", {}).get("mae") is not None]

            pred_1s = [r.prediction.get("horizon_1.0s", {}).get("mae") for r in runs if r.prediction.get("horizon_1.0s", {}).get("mae") is not None]
            pred_2s = [r.prediction.get("horizon_2.0s", {}).get("mae") for r in runs if r.prediction.get("horizon_2.0s", {}).get("mae") is not None]

            ttc_maes = [r.ttc_dcpa.get("error_stats", {}).get("mae") for r in runs if r.ttc_dcpa.get("error_stats", {}).get("mae") is not None]

            conf_rates = [r.tracking.get("confirmation_rate", 1.0) for r in runs if r.tracking.get("confirmation_rate") is not None]
            id_switches = sum(r.tracking.get("id_switch_count", 0) for r in runs)
            motas = [r.tracking.get("mota", 1.0) for r in runs if r.tracking.get("mota") is not None]

            lead_times = [r.warning.get("lead_time_s") for r in runs if r.warning.get("lead_time_s") is not None]
            false_warnings = sum(1 for r in runs if r.warning.get("false_warning_detected", False))

            def mean_or_none(lst: list[float]) -> float | None:
                return round(sum(lst) / len(lst), 4) if lst else None

            noise_summaries[cond] = {
                "run_count": n_runs,
                "mean_position_mae_m": mean_or_none(pos_maes),
                "mean_velocity_mae_mps": mean_or_none(vel_maes),
                "mean_prediction_mae_1s_m": mean_or_none(pred_1s),
                "mean_prediction_mae_2s_m": mean_or_none(pred_2s),
                "mean_ttc_mae_s": mean_or_none(ttc_maes),
                "mean_track_confirmation_rate": mean_or_none(conf_rates),
                "total_id_switches": id_switches,
                "mean_mota": mean_or_none(motas),
                "mean_warning_lead_time_s": mean_or_none(lead_times),
                "total_false_warnings": false_warnings,
            }

        return noise_summaries
