"""Comparative metric evaluation engine comparing PRISM against the Baseline."""

from __future__ import annotations

import math
from typing import Any, Sequence

from simulation.baselines.models import ReactiveBaselineDecision
from simulation.comparison.models import ComparativeAvoidanceStats, ComparativeWarningStats

PRISM_WARNING_ACTIONS = ("SLOW_DOWN", "TURN_LEFT", "TURN_RIGHT", "STOP", "EMERGENCY_STOP")
PRISM_WARNING_RISK_LEVELS = ("HIGH", "CRITICAL")


class ComparativeMetricsEvaluator:
    """Evaluates run-level and frame-level comparative performance metrics between PRISM and Baseline."""

    def __init__(self, min_safe_clearance_m: float = 0.50) -> None:
        self.min_safe_clearance_m: float = float(min_safe_clearance_m)

    def evaluate_run(
        self,
        gt_records: Sequence[dict[str, Any]],
        prism_records: Sequence[dict[str, Any]],
        baseline_decisions: Sequence[ReactiveBaselineDecision],
    ) -> tuple[ComparativeWarningStats, ComparativeAvoidanceStats]:
        """Compute paired comparative warning, avoidance, and safety statistics for a run."""
        num_frames = len(gt_records)
        dt = gt_records[1]["timestamp"] - gt_records[0]["timestamp"] if num_frames > 1 else 0.05

        first_collision_time: float | None = None
        prism_first_warn_time: float | None = None
        baseline_first_warn_time: float | None = None

        prism_actions: list[str] = []
        baseline_actions: list[str] = []
        min_clearances: list[float] = []

        for f_idx in range(num_frames):
            gt = gt_records[f_idx]
            pr = prism_records[f_idx]
            bl = baseline_decisions[f_idx]
            t = gt["timestamp"]

            # 1. Ground Truth Collision
            if gt.get("collision_state", {}).get("has_collided", False) and first_collision_time is None:
                first_collision_time = t

            # 2. PRISM Warning Alert
            pr_act = pr.get("decision_action", "CONTINUE")
            pr_risk = pr.get("risk_level", "SAFE")
            pr_is_warn = (pr_act in PRISM_WARNING_ACTIONS or pr_risk in PRISM_WARNING_RISK_LEVELS)
            if pr_is_warn and prism_first_warn_time is None:
                prism_first_warn_time = t
            prism_actions.append(pr_act)

            # 3. Baseline Warning Alert
            bl_is_warn = bl.is_warning
            if bl_is_warn and baseline_first_warn_time is None:
                baseline_first_warn_time = t
            baseline_actions.append(bl.action)

            # 4. True Physical Clearance
            ego = gt.get("ego", {})
            ego_x, ego_z = ego.get("x", 0.0), ego.get("z", 0.0)
            ego_r = ego.get("radius", 0.35)

            frame_min_clr = 999.0
            for obj in gt.get("objects", []):
                dx = obj["x"] - ego_x
                dz = obj["z"] - ego_z
                clr = math.sqrt(dx * dx + dz * dz) - (ego_r + obj.get("radius", 0.35))
                if clr < frame_min_clr:
                    frame_min_clr = clr

            if frame_min_clr < 990.0:
                min_clearances.append(frame_min_clr)

        has_collision = (first_collision_time is not None)
        prism_lead_s: float | None = None
        baseline_lead_s: float | None = None
        delta_lead_s: float | None = None
        earlier_system: str | None = None
        prism_maint_ratio: float | None = None
        baseline_maint_ratio: float | None = None
        prism_false_warn = False
        baseline_false_warn = False

        if has_collision:
            # PRISM Lead Time
            if prism_first_warn_time is not None and prism_first_warn_time <= first_collision_time:
                prism_lead_s = round(first_collision_time - prism_first_warn_time, 4)
                f_s = int(round(prism_first_warn_time / dt))
                f_e = int(round(first_collision_time / dt))
                span = max(1, f_e - f_s + 1)
                active_pr = sum(
                    1 for k in range(f_s, min(num_frames, f_e + 1))
                    if (prism_records[k].get("decision_action") in PRISM_WARNING_ACTIONS
                        or prism_records[k].get("risk_level") in PRISM_WARNING_RISK_LEVELS)
                )
                prism_maint_ratio = round(active_pr / span, 4)

            # Baseline Lead Time
            if baseline_first_warn_time is not None and baseline_first_warn_time <= first_collision_time:
                baseline_lead_s = round(first_collision_time - baseline_first_warn_time, 4)
                f_s = int(round(baseline_first_warn_time / dt))
                f_e = int(round(first_collision_time / dt))
                span = max(1, f_e - f_s + 1)
                active_bl = sum(
                    1 for k in range(f_s, min(num_frames, f_e + 1))
                    if baseline_decisions[k].is_warning
                )
                baseline_maint_ratio = round(active_bl / span, 4)

            # Paired Difference & Earlier System
            if prism_lead_s is not None and baseline_lead_s is not None:
                delta_lead_s = round(prism_lead_s - baseline_lead_s, 4)
                if delta_lead_s > 1e-4:
                    earlier_system = "PRISM"
                elif delta_lead_s < -1e-4:
                    earlier_system = "BASELINE"
                else:
                    earlier_system = "EQUAL"
            elif prism_lead_s is not None and baseline_lead_s is None:
                earlier_system = "PRISM"
            elif prism_lead_s is None and baseline_lead_s is not None:
                earlier_system = "BASELINE"
            else:
                earlier_system = "NONE"
        else:
            # Safe run: evaluate false warning triggers
            prism_false_warn = (prism_first_warn_time is not None)
            baseline_false_warn = (baseline_first_warn_time is not None)

        warning_stats = ComparativeWarningStats(
            has_collision=has_collision,
            prism_warned=(prism_first_warn_time is not None),
            baseline_warned=(baseline_first_warn_time is not None),
            prism_first_warning_time_s=prism_first_warn_time,
            baseline_first_warning_time_s=baseline_first_warn_time,
            collision_time_s=first_collision_time,
            prism_lead_time_s=prism_lead_s,
            baseline_lead_time_s=baseline_lead_s,
            delta_lead_time_s=delta_lead_s,
            earlier_warning_system=earlier_system,
            prism_maintained_ratio=prism_maint_ratio,
            baseline_maintained_ratio=baseline_maint_ratio,
            prism_false_warning=prism_false_warn,
            baseline_false_warning=baseline_false_warn,
        )

        # 5. Avoidance Stability & Clearance Metrics
        def compute_action_stability(actions: list[str]) -> tuple[int, int]:
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
            return changes, reversals

        pr_changes, pr_reversals = compute_action_stability(prism_actions)
        bl_changes, bl_reversals = compute_action_stability(baseline_actions)

        min_clr_val = min(min_clearances) if min_clearances else 0.0
        safe_clrs = sum(1 for c in min_clearances if c >= self.min_safe_clearance_m)
        safe_clr_ratio = safe_clrs / max(1, len(min_clearances))

        avoidance_stats = ComparativeAvoidanceStats(
            prism_decision_changes=pr_changes,
            baseline_decision_changes=bl_changes,
            prism_steering_reversals=pr_reversals,
            baseline_steering_reversals=bl_reversals,
            prism_min_clearance_m=round(min_clr_val, 4),
            baseline_min_clearance_m=round(min_clr_val, 4),
            prism_safe_clearance_ratio=round(safe_clr_ratio, 4),
            baseline_safe_clearance_ratio=round(safe_clr_ratio, 4),
        )

        return warning_stats, avoidance_stats
