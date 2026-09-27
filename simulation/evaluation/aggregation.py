"""Macro and micro aggregation engine for PRISM Part 17 evaluation."""

from __future__ import annotations

import datetime
from typing import Any, Sequence

from simulation.evaluation.models import (
    AggregateEvaluationReport,
    MetricStats,
    RiskCalibrationBin,
    RunEvaluationResult,
)
from simulation.evaluation.state_metrics import compute_metric_stats


class MetricsAggregator:
    """Aggregates individual run evaluation results into macro/micro summaries and benchmark reports."""

    def aggregate_dataset(
        self,
        dataset_name: str,
        run_results: Sequence[RunEvaluationResult],
        evaluation_version: str = "17.0.0",
    ) -> AggregateEvaluationReport:
        """Compute aggregate evaluation metrics across all runs in a dataset."""
        if not run_results:
            now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
            return AggregateEvaluationReport(
                dataset_name=dataset_name,
                evaluation_version=evaluation_version,
                created_at=now_str,
                total_runs_evaluated=0,
                total_frames_evaluated=0,
                scenario_families_evaluated=[],
                noise_conditions_evaluated=[],
                overall_state_estimation={},
                overall_tracking={},
                overall_prediction={},
                overall_ttc_dcpa={},
                overall_risk={},
                overall_warning={},
                overall_avoidance={},
                scenario_breakdown={},
                noise_breakdown={},
                risk_calibration_bins=[],
                total_failures_identified=0,
            )

        total_frames = sum(r.num_frames for r in run_results)
        scenario_families = sorted(list({r.scenario_family for r in run_results}))
        noise_conditions = sorted(list({r.noise_condition for r in run_results}))
        total_failures = sum(len(r.failures) for r in run_results)

        # 1. State Estimation Aggregates
        pos_maes = [r.state_estimation["position_2d"]["mae"] for r in run_results if r.state_estimation["position_2d"]["mae"] is not None]
        pos_rmses = [r.state_estimation["position_2d"]["rmse"] for r in run_results if r.state_estimation["position_2d"]["rmse"] is not None]
        vel_maes = [r.state_estimation["velocity_2d"]["mae"] for r in run_results if r.state_estimation["velocity_2d"]["mae"] is not None]
        vel_rmses = [r.state_estimation["velocity_2d"]["rmse"] for r in run_results if r.state_estimation["velocity_2d"]["rmse"] is not None]

        overall_state = {
            "position_2d_macro": compute_metric_stats(pos_maes).to_dict(),
            "position_rmse_macro": compute_metric_stats(pos_rmses).to_dict(),
            "velocity_2d_macro": compute_metric_stats(vel_maes).to_dict(),
            "velocity_rmse_macro": compute_metric_stats(vel_rmses).to_dict(),
        }

        # 2. Tracking Aggregates
        conf_rates = [r.tracking["confirmation_rate"] for r in run_results]
        id_switches_total = sum(r.tracking["id_switch_count"] for r in run_results)
        frag_total = sum(r.tracking["fragmentation_count"] for r in run_results)
        mota_vals = [r.tracking["mota"] for r in run_results]
        idf1_vals = [r.tracking["idf1"] for r in run_results]

        overall_tracking = {
            "mean_confirmation_rate": round(sum(conf_rates) / len(conf_rates), 4),
            "total_id_switches": id_switches_total,
            "total_fragmentations": frag_total,
            "mean_mota": round(sum(mota_vals) / len(mota_vals), 4),
            "mean_idf1": round(sum(idf1_vals) / len(idf1_vals), 4),
        }

        # 3. Prediction Aggregates
        overall_pred: dict[str, Any] = {}
        for h_key in ("horizon_0.5s", "horizon_1.0s", "horizon_2.0s", "horizon_3.0s"):
            maes = [r.prediction.get(h_key, {}).get("mae") for r in run_results if r.prediction.get(h_key, {}).get("mae") is not None]
            overall_pred[h_key] = compute_metric_stats(maes).to_dict()

        # 4. TTC & dCPA Aggregates
        ttc_maes = [r.ttc_dcpa["error_stats"]["mae"] for r in run_results if r.ttc_dcpa["error_stats"]["mae"] is not None]
        dcpa_maes = [r.ttc_dcpa.get("dcpa_stats", {}).get("mae") for r in run_results if r.ttc_dcpa.get("dcpa_stats", {}).get("mae") is not None]

        overall_ttc_dcpa = {
            "ttc_mae_macro": compute_metric_stats(ttc_maes).to_dict(),
            "dcpa_mae_macro": compute_metric_stats(dcpa_maes).to_dict(),
            "total_finite_vs_finite_samples": sum(r.ttc_dcpa.get("finite_vs_finite_count", 0) for r in run_results),
            "total_false_predicted_collisions": sum(r.ttc_dcpa.get("false_predicted_collision_count", 0) for r in run_results),
            "total_missed_predicted_collisions": sum(r.ttc_dcpa.get("missed_predicted_collision_count", 0) for r in run_results),
        }

        # 5. Risk Multi-Horizon Detection Aggregates
        overall_risk: dict[str, Any] = {}
        for h_key in ("horizon_1.0s", "horizon_2.0s", "horizon_3.0s"):
            tps = sum(r.risk["multi_horizon_detection"].get(h_key, {}).get("tp", 0) for r in run_results)
            tns = sum(r.risk["multi_horizon_detection"].get(h_key, {}).get("tn", 0) for r in run_results)
            fps = sum(r.risk["multi_horizon_detection"].get(h_key, {}).get("fp", 0) for r in run_results)
            fns = sum(r.risk["multi_horizon_detection"].get(h_key, {}).get("fn", 0) for r in run_results)

            prec = tps / (tps + fps) if (tps + fps) > 0 else (1.0 if tps == 0 and fps == 0 else 0.0)
            rec = tps / (tps + fns) if (tps + fns) > 0 else (1.0 if tps == 0 and fns == 0 else 0.0)
            f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            overall_risk[h_key] = {
                "tp": tps,
                "tn": tns,
                "fp": fps,
                "fn": fns,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }

        # 6. Warning Lead Time & Event-Level Aggregates
        lead_times = [r.warning["lead_time_s"] for r in run_results if r.warning["lead_time_s"] is not None]
        false_warn_count = sum(1 for r in run_results if r.warning["false_warning_detected"])
        collision_runs = sum(1 for r in run_results if r.warning["has_collision_event"])
        safe_runs = len(run_results) - collision_runs

        tp_count = sum(1 for r in run_results if r.warning["has_collision_event"] and r.warning["lead_time_s"] is not None and r.warning["lead_time_s"] > 0)
        fn_count = sum(1 for r in run_results if r.warning["has_collision_event"] and (r.warning["lead_time_s"] is None or r.warning["lead_time_s"] <= 0))
        fp_count = sum(1 for r in run_results if (not r.warning["has_collision_event"]) and r.warning["false_warning_detected"])
        tn_count = sum(1 for r in run_results if (not r.warning["has_collision_event"]) and (not r.warning["false_warning_detected"]))

        event_prec = round(tp_count / (tp_count + fp_count), 4) if (tp_count + fp_count) > 0 else 0.0
        event_rec = round(tp_count / (tp_count + fn_count), 4) if (tp_count + fn_count) > 0 else 0.0
        event_f1 = round((2.0 * event_prec * event_rec) / (event_prec + event_rec), 4) if (event_prec + event_rec) > 0 else 0.0
        event_miss_rate = round(fn_count / (tp_count + fn_count), 4) if (tp_count + fn_count) > 0 else 0.0

        overall_warning = {
            "lead_time_stats": compute_metric_stats(lead_times).to_dict(),
            "collision_run_count": collision_runs,
            "safe_run_count": safe_runs,
            "false_warning_count": false_warn_count,
            "false_warning_rate_on_safe_runs": round(false_warn_count / max(1, safe_runs), 4),
            "event_level_metrics": {
                "tp": tp_count,
                "fn": fn_count,
                "fp": fp_count,
                "tn": tn_count,
                "precision": event_prec,
                "recall": event_rec,
                "f1_score": event_f1,
                "miss_rate": event_miss_rate,
            },
        }

        # 7. Avoidance Aggregates
        dec_changes = [r.avoidance["decision_change_count"] for r in run_results]
        reversals = sum(r.avoidance["steering_reversal_count"] for r in run_results)
        safe_ratios = [r.avoidance["safe_clearance_ratio"] for r in run_results]

        overall_avoidance = {
            "mean_decision_changes_per_run": round(sum(dec_changes) / len(dec_changes), 2),
            "total_steering_reversals": reversals,
            "mean_safe_clearance_ratio": round(sum(safe_ratios) / len(safe_ratios), 4),
        }

        # 8. Scenario Breakdown
        scenario_breakdown: dict[str, Any] = {}
        for fam in scenario_families:
            fam_runs = [r for r in run_results if r.scenario_family == fam]
            f_pos_maes = [r.state_estimation["position_2d"]["mae"] for r in fam_runs if r.state_estimation["position_2d"]["mae"] is not None]
            f_lead_times = [r.warning["lead_time_s"] for r in fam_runs if r.warning["lead_time_s"] is not None]
            f_conf = [r.tracking["confirmation_rate"] for r in fam_runs]
            f_switches = sum(r.tracking["id_switch_count"] for r in fam_runs)

            scenario_breakdown[fam] = {
                "run_count": len(fam_runs),
                "position_mae_m": round(sum(f_pos_maes) / len(f_pos_maes), 4) if f_pos_maes else None,
                "mean_confirmation_rate": round(sum(f_conf) / len(f_conf), 4),
                "total_id_switches": f_switches,
                "mean_lead_time_s": round(sum(f_lead_times) / len(f_lead_times), 4) if f_lead_times else None,
                "failures_count": sum(len(r.failures) for r in fam_runs),
            }

        # 9. Aggregate Risk Calibration Bins
        aggregated_bins: list[dict[str, Any]] = []
        if run_results:
            first_bins = run_results[0].risk.get("risk_calibration_bins", [])
            for b_idx, b_proto in enumerate(first_bins):
                b_label = b_proto["bin_label"]
                total_samples = sum(r.risk["risk_calibration_bins"][b_idx]["sample_count"] for r in run_results)
                total_events = sum(r.risk["risk_calibration_bins"][b_idx]["collision_event_count"] for r in run_results)
                freq = round(total_events / total_samples, 4) if total_samples > 0 else None
                aggregated_bins.append(
                    {
                        "bin_label": b_label,
                        "min_score": b_proto["min_score"],
                        "max_score": b_proto["max_score"],
                        "total_samples": total_samples,
                        "collision_events": total_events,
                        "empirical_collision_frequency": freq,
                    }
                )

        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return AggregateEvaluationReport(
            dataset_name=dataset_name,
            evaluation_version=evaluation_version,
            created_at=now_str,
            total_runs_evaluated=len(run_results),
            total_frames_evaluated=total_frames,
            scenario_families_evaluated=scenario_families,
            noise_conditions_evaluated=noise_conditions,
            overall_state_estimation=overall_state,
            overall_tracking=overall_tracking,
            overall_prediction=overall_pred,
            overall_ttc_dcpa=overall_ttc_dcpa,
            overall_risk=overall_risk,
            overall_warning=overall_warning,
            overall_avoidance=overall_avoidance,
            scenario_breakdown=scenario_breakdown,
            noise_breakdown={},
            risk_calibration_bins=aggregated_bins,
            total_failures_identified=total_failures,
        )
