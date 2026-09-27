"""Automated diagnostic failure extraction engine for PRISM Part 17."""

from __future__ import annotations

import math
from typing import Any, Sequence

from simulation.evaluation.alignment import AlignedFrame
from simulation.evaluation.models import EvaluationConfig, FailureCase


class FailureAnalyzer:
    """Detects and extracts structured diagnostic failure cases across experimental runs."""

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config: EvaluationConfig = config if config is not None else EvaluationConfig()

    def extract_failures(
        self,
        scenario_id: str,
        scenario_family: str,
        run_id: str,
        aligned_frames: Sequence[AlignedFrame],
    ) -> list[FailureCase]:
        """Inspect an aligned run and return structured records for any detected diagnostic failures."""
        failures: list[FailureCase] = []

        num_frames = len(aligned_frames)
        dt = aligned_frames[1].timestamp - aligned_frames[0].timestamp if num_frames > 1 else 0.05
        pred_thresh = self.config.diagnostic_pred_error_threshold_m
        ttc_thresh = self.config.diagnostic_ttc_error_threshold_s

        # Track ID continuity per GT object
        gt_last_track: dict[Any, int | None] = {}

        for f_idx, frame in enumerate(aligned_frames):
            t = frame.timestamp
            gt_col = frame.gt_collision_state
            gt_ttc = gt_col.get("time_to_collision")
            pr_out = frame.prism_output
            pr_ttc = pr_out.get("predicted_ttc")
            pr_risk_lvl = pr_out.get("risk_level", "SAFE")

            # 1. TTC Diagnostic Failure
            gt_finite = (gt_ttc is not None and math.isfinite(gt_ttc))
            pr_finite = (pr_ttc is not None and math.isfinite(pr_ttc))

            if gt_finite and pr_finite:
                err = abs(pr_ttc - gt_ttc)
                if err > ttc_thresh:
                    failures.append(
                        FailureCase(
                            failure_type="TTC_ESTIMATION_ERROR",
                            scenario_id=scenario_id,
                            scenario_family=scenario_family,
                            run_id=run_id,
                            timestamp=t,
                            frame_index=f_idx,
                            object_id=gt_col.get("colliding_agent_id"),
                            track_id=pr_out.get("primary_threat_id"),
                            metric_name="abs_ttc_error_s",
                            metric_value=round(err, 3),
                            ground_truth_summary={"gt_ttc": gt_ttc, "gt_dcpa": gt_col.get("closest_point_of_approach")},
                            prism_output_summary={"pred_ttc": pr_ttc, "pred_cpa": pr_out.get("predicted_cpa")},
                            description=f"TTC error ({err:.2f}s) exceeded diagnostic threshold ({ttc_thresh:.2f}s).",
                        )
                    )
            elif gt_finite and not pr_finite:
                # Missed collision prediction
                failures.append(
                    FailureCase(
                        failure_type="MISSED_COLLISION_PREDICTION",
                        scenario_id=scenario_id,
                        scenario_family=scenario_family,
                        run_id=run_id,
                        timestamp=t,
                        frame_index=f_idx,
                        object_id=gt_col.get("colliding_agent_id"),
                        track_id=pr_out.get("primary_threat_id"),
                        metric_name="predicted_ttc",
                        metric_value=None,
                        ground_truth_summary={"gt_ttc": gt_ttc},
                        prism_output_summary={"predicted_ttc": None},
                        description="Ground truth has finite collision TTC but PRISM predicted None/infinity.",
                    )
                )

            # 2. Risk False Negatives (Imminent true collision within 2.0s but risk is SAFE/LOW)
            steps_2s = int(round(2.0 / dt))
            max_step = min(num_frames, f_idx + steps_2s + 1)
            future_collision_2s = any(aligned_frames[k].gt_collision_state.get("has_collided", False) for k in range(f_idx, max_step))

            if future_collision_2s and pr_risk_lvl in ("SAFE", "LOW"):
                failures.append(
                    FailureCase(
                        failure_type="RISK_FALSE_NEGATIVE",
                        scenario_id=scenario_id,
                        scenario_family=scenario_family,
                        run_id=run_id,
                        timestamp=t,
                        frame_index=f_idx,
                        object_id=None,
                        track_id=pr_out.get("primary_threat_id"),
                        metric_name="risk_level",
                        metric_value=pr_risk_lvl,
                        ground_truth_summary={"collision_within_2s": True},
                        prism_output_summary={"risk_level": pr_risk_lvl, "risk_score": pr_out.get("risk_score")},
                        description="True collision imminent within 2.0s but PRISM evaluated risk as SAFE/LOW.",
                    )
                )

            # 3. Tracking ID Switches
            for gt_obj, trk in frame.matched_pairs:
                g_id = gt_obj["object_id"]
                t_id = trk["track_id"]
                if g_id in gt_last_track:
                    prev_t = gt_last_track[g_id]
                    if prev_t is not None and t_id != prev_t:
                        failures.append(
                            FailureCase(
                                failure_type="TRACKING_ID_SWITCH",
                                scenario_id=scenario_id,
                                scenario_family=scenario_family,
                                run_id=run_id,
                                timestamp=t,
                                frame_index=f_idx,
                                object_id=g_id,
                                track_id=t_id,
                                metric_name="previous_track_id",
                                metric_value=prev_t,
                                ground_truth_summary={"gt_id": g_id},
                                prism_output_summary={"new_track_id": t_id, "prev_track_id": prev_t},
                                description=f"Track ID switched from #{prev_t} to #{t_id} for GT object #{g_id}.",
                            )
                        )
                gt_last_track[g_id] = t_id

        return failures
