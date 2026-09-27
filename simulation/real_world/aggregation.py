"""Statistical aggregation engine for PRISM Part 20 Real-World Benchmark."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from simulation.real_world.models import RealWorldRunResult


def compute_mean(values: Sequence[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    return (sum(valid) / len(valid)) if valid else None


def compute_median(values: Sequence[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    s = sorted(valid)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def compute_std(values: Sequence[float | None], ddof: int = 1) -> float | None:
    valid = [v for v in values if v is not None]
    n = len(valid)
    if n <= ddof:
        return 0.0 if n > 0 else None
    mean = sum(valid) / n
    var = sum((x - mean) ** 2 for x in valid) / (n - ddof)
    return math.sqrt(max(0.0, var))


class RealWorldAggregator:
    """Aggregates real-world benchmark metrics across runs and scenario families."""

    def __init__(self) -> None:
        pass

    def compile_summary(self, results: Sequence[RealWorldRunResult]) -> dict[str, Any]:
        """Compile overall statistical summary across a collection of runs."""
        total_runs = len(results)
        collision_runs = [r for r in results if r.has_collision]
        safe_runs = [r for r in results if not r.has_collision]

        n_col = len(collision_runs)
        n_safe = len(safe_runs)

        lead_times = [r.warning_lead_time_s for r in collision_runs if r.warning_lead_time_s is not None]
        recalled_col_count = sum(1 for r in collision_runs if (r.collision_recalled or r.warned))
        missed_col_count = sum(1 for r in collision_runs if (r.collision_missed or not r.warned))
        prox_caution_count = sum(1 for r in safe_runs if (r.proximity_caution_alert or r.caution_alert_issued))
        false_col_warn_count = sum(1 for r in safe_runs if (r.false_collision_warning or r.emergency_warning_issued))
        false_warn_count = sum(1 for r in safe_runs if r.false_warning)

        continuities = [r.tracking_continuity_ratio for r in results]
        track_coverages = [r.track_coverage_ratio for r in results]
        det_assoc_coverages = [r.detection_association_coverage_ratio for r in results]
        zero_det_frames_list = [r.zero_detection_frames for r in results]
        coasting_frames_list = [r.coasting_frames for r in results]
        gap_bridging_rates = [r.gap_bridging_rate for r in results]
        decision_changes = [float(r.decision_changes) for r in results]
        steering_reversals = [float(r.steering_reversals) for r in results]
        latencies = [r.mean_latency_ms for r in results]
        det_counts = [r.mean_detection_count for r in results]

        mean_lead = compute_mean(lead_times)
        median_lead = compute_median(lead_times)
        std_lead = compute_std(lead_times)
        mean_cont = compute_mean(continuities)
        mean_trk_cov = compute_mean(track_coverages)
        mean_assoc_cov = compute_mean(det_assoc_coverages)
        mean_gap_bridge = compute_mean(gap_bridging_rates)
        mean_det = compute_mean(det_counts)
        mean_dec_chg = compute_mean(decision_changes)
        mean_steer_rev = compute_mean(steering_reversals)
        mean_lat = compute_mean(latencies)

        col_recall = (recalled_col_count / n_col) if n_col > 0 else None
        col_miss = (missed_col_count / n_col) if n_col > 0 else None
        prox_caution_rate = (prox_caution_count / n_safe) if n_safe > 0 else None
        false_col_warn_rate = (false_col_warn_count / n_safe) if n_safe > 0 else None

        # Backwards compatible alias rates
        collision_warn_rate = col_recall
        safe_false_warn_rate = (false_warn_count / n_safe) if n_safe > 0 else None

        return {
            "total_runs": total_runs,
            "collision_runs": n_col,
            "safe_runs": n_safe,
            "collision_event_recall": round(col_recall, 4) if col_recall is not None else None,
            "collision_event_miss_rate": round(col_miss, 4) if col_miss is not None else None,
            "proximity_caution_alert_rate": round(prox_caution_rate, 4) if prox_caution_rate is not None else None,
            "false_collision_warning_rate": round(false_col_warn_rate, 4) if false_col_warn_rate is not None else None,
            "collision_warning_rate": round(collision_warn_rate, 4) if collision_warn_rate is not None else None,
            "safe_false_warning_rate": round(safe_false_warn_rate, 4) if safe_false_warn_rate is not None else None,
            "mean_warning_lead_time_s": round(mean_lead, 4) if mean_lead is not None else None,
            "median_warning_lead_time_s": round(median_lead, 4) if median_lead is not None else None,
            "std_warning_lead_time_s": round(std_lead, 4) if std_lead is not None else None,
            "mean_tracking_continuity": round(mean_cont, 4) if mean_cont is not None else None,
            "mean_track_coverage": round(mean_trk_cov, 4) if mean_trk_cov is not None else None,
            "mean_detection_association_coverage": round(mean_assoc_cov, 4) if mean_assoc_cov is not None else None,
            "total_zero_detection_frames": sum(zero_det_frames_list),
            "total_detector_dropout_frames": sum(zero_det_frames_list),
            "total_coasting_frames": sum(coasting_frames_list),
            "mean_gap_bridging_rate": round(mean_gap_bridge, 4) if mean_gap_bridge is not None else None,
            "mean_dropout_bridging_rate": round(mean_gap_bridge, 4) if mean_gap_bridge is not None else None,
            "mean_detection_count": round(mean_det, 2) if mean_det is not None else None,
            "mean_decision_changes": round(mean_dec_chg, 2) if mean_dec_chg is not None else None,
            "mean_steering_reversals": round(mean_steer_rev, 2) if mean_steer_rev is not None else None,
            "mean_latency_ms": round(mean_lat, 3) if mean_lat is not None else None,
        }

    def aggregate_by_scenario(self, results: Sequence[RealWorldRunResult]) -> dict[str, Any]:
        """Group and summarize results by scenario family (R01 to R06)."""
        families = sorted(list({r.scenario_family for r in results}))
        scenario_report: dict[str, Any] = {}

        for fam in families:
            fam_runs = [r for r in results if r.scenario_family == fam]
            scenario_report[fam] = self.compile_summary(fam_runs)

        return scenario_report

