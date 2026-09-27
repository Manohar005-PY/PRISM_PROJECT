"""Evaluation metrics calculation for PRISM Part 19 Ablation & Sensitivity Analysis."""

from __future__ import annotations

import math
from typing import Any, Sequence

from simulation.ablation.models import (
    PairwiseVariantComparison,
    VariantFrameOutput,
    VariantRunResult,
)

WARNING_ACTIONS = ("SLOW_DOWN", "TURN_LEFT", "TURN_RIGHT", "STOP", "EMERGENCY_STOP")
WARNING_RISK_LEVELS = ("HIGH", "CRITICAL")


class AblationMetricsEvaluator:
    """Computes run-level and paired comparative ablation metrics."""

    def __init__(self, min_safe_clearance_m: float = 0.50) -> None:
        self.min_safe_clearance_m: float = float(min_safe_clearance_m)

    def evaluate_variant_run(
        self,
        metadata: dict[str, Any],
        gt_records: Sequence[dict[str, Any]],
        frame_outputs: Sequence[VariantFrameOutput],
        variant_id: str,
    ) -> VariantRunResult:
        """Compute full summary metrics for a single variant on an experimental run."""
        scenario_id = metadata.get("scenario_id", "unknown_scenario")
        scenario_family = metadata.get("scenario_family", "unknown_family")
        run_id = metadata.get("run_id", scenario_id)
        noise_cond = metadata.get("parameters", {}).get("noise_level", "NO_NOISE")
        duration = float(metadata.get("duration", 0.0))
        num_frames = len(gt_records)
        dt = gt_records[1]["timestamp"] - gt_records[0]["timestamp"] if num_frames > 1 else 0.05

        first_collision_time: float | None = None
        first_warn_time: float | None = None
        actions: list[str] = []
        min_clearances: list[float] = []
        latencies: list[float] = []

        for f_idx in range(num_frames):
            gt = gt_records[f_idx]
            out = frame_outputs[f_idx] if f_idx < len(frame_outputs) else None
            t = float(gt["timestamp"])

            # 1. Collision check
            if gt.get("collision_state", {}).get("has_collided", False) and first_collision_time is None:
                first_collision_time = t

            # 2. Warning trigger check
            if out is not None:
                actions.append(out.action)
                latencies.append(out.latency_ms)
                is_warn = (out.action in WARNING_ACTIONS or out.risk_level in WARNING_RISK_LEVELS)
                if is_warn and first_warn_time is None:
                    first_warn_time = t
            else:
                actions.append("CONTINUE")

            # 3. Ground truth clearance calculation
            ego = gt.get("ego", {})
            ego_x, ego_z = float(ego.get("x", 0.0)), float(ego.get("z", 0.0))
            ego_r = float(ego.get("radius", 0.35))

            frame_min_clr = 999.0
            for obj in gt.get("objects", []):
                dx = float(obj["x"]) - ego_x
                dz = float(obj["z"]) - ego_z
                clr = math.sqrt(dx * dx + dz * dz) - (ego_r + float(obj.get("radius", 0.35)))
                if clr < frame_min_clr:
                    frame_min_clr = clr

            if frame_min_clr < 990.0:
                min_clearances.append(frame_min_clr)

        has_collision = (first_collision_time is not None)
        lead_time_s: float | None = None
        maint_ratio: float | None = None
        false_warn = False

        if has_collision:
            if first_warn_time is not None and first_warn_time <= first_collision_time:
                lead_time_s = round(first_collision_time - first_warn_time, 4)
                f_s = int(round(first_warn_time / dt))
                f_e = int(round(first_collision_time / dt))
                span = max(1, f_e - f_s + 1)
                active_warns = sum(
                    1 for k in range(f_s, min(num_frames, f_e + 1))
                    if k < len(frame_outputs) and (
                        frame_outputs[k].action in WARNING_ACTIONS
                        or frame_outputs[k].risk_level in WARNING_RISK_LEVELS
                    )
                )
                maint_ratio = round(active_warns / span, 4)
        else:
            false_warn = (first_warn_time is not None)

        # Decision changes and steering reversals
        changes = 0
        reversals = 0
        prev_act: str | None = None
        prev_turn: str | None = None
        for a in actions:
            if prev_act is None:
                prev_act = a
            elif a != prev_act:
                changes += 1
                prev_act = a
            if a in ("TURN_LEFT", "TURN_RIGHT"):
                if prev_turn is not None and a != prev_turn:
                    reversals += 1
                prev_turn = a

        min_clr_val = min(min_clearances) if min_clearances else 0.0
        safe_clrs = sum(1 for c in min_clearances if c >= self.min_safe_clearance_m)
        safe_ratio = round(safe_clrs / max(1, len(min_clearances)), 4)
        mean_lat = round(sum(latencies) / max(1, len(latencies)), 3)

        return VariantRunResult(
            scenario_id=scenario_id,
            scenario_family=scenario_family,
            run_id=run_id,
            variant_id=variant_id,
            num_frames=num_frames,
            duration=duration,
            noise_condition=noise_cond,
            has_collision=has_collision,
            collision_time_s=first_collision_time,
            first_warning_time_s=first_warn_time,
            warning_lead_time_s=lead_time_s,
            warned=(first_warn_time is not None),
            false_warning=false_warn,
            warning_maintained_ratio=maint_ratio,
            decision_changes=changes,
            steering_reversals=reversals,
            min_clearance_m=round(min_clr_val, 4),
            safe_clearance_ratio=safe_ratio,
            mean_latency_ms=mean_lat,
            frame_outputs=list(frame_outputs),
        )

    def compare_runs(
        self,
        full_res: VariantRunResult,
        variant_res: VariantRunResult,
    ) -> PairwiseVariantComparison:
        """Compute paired differences between FULL_PRISM and an ablated variant for one run."""
        delta_lead: float | None = None
        if full_res.warning_lead_time_s is not None and variant_res.warning_lead_time_s is not None:
            delta_lead = round(full_res.warning_lead_time_s - variant_res.warning_lead_time_s, 4)
        elif full_res.warning_lead_time_s is not None and variant_res.warning_lead_time_s is None:
            delta_lead = round(full_res.warning_lead_time_s, 4)
        elif full_res.warning_lead_time_s is None and variant_res.warning_lead_time_s is not None:
            delta_lead = round(-variant_res.warning_lead_time_s, 4)

        delta_clr = round(full_res.min_clearance_m - variant_res.min_clearance_m, 4)
        delta_changes = variant_res.decision_changes - full_res.decision_changes
        delta_reversals = variant_res.steering_reversals - full_res.steering_reversals

        return PairwiseVariantComparison(
            scenario_id=full_res.scenario_id,
            scenario_family=full_res.scenario_family,
            run_id=full_res.run_id,
            variant_id=variant_res.variant_id,
            noise_condition=full_res.noise_condition,
            has_collision=full_res.has_collision,
            full_lead_time_s=full_res.warning_lead_time_s,
            variant_lead_time_s=variant_res.warning_lead_time_s,
            delta_lead_time_s=delta_lead,
            full_min_clearance_m=full_res.min_clearance_m,
            variant_min_clearance_m=variant_res.min_clearance_m,
            delta_min_clearance_m=delta_clr,
            full_decision_changes=full_res.decision_changes,
            variant_decision_changes=variant_res.decision_changes,
            delta_decision_changes=delta_changes,
            full_steering_reversals=full_res.steering_reversals,
            variant_steering_reversals=variant_res.steering_reversals,
            delta_steering_reversals=delta_reversals,
            full_warned=full_res.warned,
            variant_warned=variant_res.warned,
            full_false_warning=full_res.false_warning,
            variant_false_warning=variant_res.false_warning,
        )
