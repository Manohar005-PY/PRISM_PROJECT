"""Automated comparative failure mode identification for PRISM Part 19."""

from __future__ import annotations

from typing import Sequence

from simulation.ablation.models import (
    AblationConfig,
    AblationFailureCase,
    PairwiseVariantComparison,
    VariantRunResult,
)


class AblationFailureAnalyzer:
    """Classifies and isolates comparative failure cases between FULL_PRISM and ablated variants."""

    def __init__(self, config: AblationConfig | None = None) -> None:
        self.config: AblationConfig = config if config is not None else AblationConfig()

    def analyze_comparison(
        self,
        full_res: VariantRunResult,
        variant_res: VariantRunResult,
        pairwise: PairwiseVariantComparison,
    ) -> list[AblationFailureCase]:
        """Detect and classify all comparative failure modes for a single run comparison."""
        failures: list[AblationFailureCase] = []

        sc_id = full_res.scenario_id
        fam = full_res.scenario_family
        run_id = full_res.run_id
        var_id = variant_res.variant_id

        # 1. Missed Warning (Critical Collision Safety Failure)
        if full_res.has_collision and full_res.warned and not variant_res.warned:
            failures.append(
                AblationFailureCase(
                    scenario_id=sc_id,
                    scenario_family=fam,
                    run_id=run_id,
                    variant_id=var_id,
                    failure_type="MISSED_WARNING",
                    description=f"Variant {var_id} failed to warn before collision at t={full_res.collision_time_s}s, while FULL_PRISM warned with lead time {full_res.warning_lead_time_s}s.",
                    timestamp_s=full_res.collision_time_s,
                    details={
                        "full_lead_time_s": full_res.warning_lead_time_s,
                        "variant_warned": False,
                    },
                )
            )

        # 2. Late Warning (Predictive Advantage Degradation)
        if (
            full_res.has_collision
            and pairwise.delta_lead_time_s is not None
            and pairwise.delta_lead_time_s >= self.config.lead_time_late_threshold_s
        ):
            failures.append(
                AblationFailureCase(
                    scenario_id=sc_id,
                    scenario_family=fam,
                    run_id=run_id,
                    variant_id=var_id,
                    failure_type="LATE_WARNING",
                    description=f"Variant {var_id} warned {pairwise.delta_lead_time_s}s later than FULL_PRISM (Full: {full_res.warning_lead_time_s}s vs Variant: {variant_res.warning_lead_time_s}s).",
                    timestamp_s=variant_res.first_warning_time_s,
                    details={
                        "delta_lead_time_s": pairwise.delta_lead_time_s,
                        "full_lead_time_s": full_res.warning_lead_time_s,
                        "variant_lead_time_s": variant_res.warning_lead_time_s,
                    },
                )
            )

        # 3. False Warning (Over-conservatism in Safe Conditions)
        if (not full_res.has_collision) and variant_res.false_warning and (not full_res.false_warning):
            failures.append(
                AblationFailureCase(
                    scenario_id=sc_id,
                    scenario_family=fam,
                    run_id=run_id,
                    variant_id=var_id,
                    failure_type="FALSE_WARNING",
                    description=f"Variant {var_id} triggered a false warning at t={variant_res.first_warning_time_s}s in a non-colliding run where FULL_PRISM remained quiet.",
                    timestamp_s=variant_res.first_warning_time_s,
                    details={
                        "variant_first_warn_time": variant_res.first_warning_time_s,
                    },
                )
            )

        # 4. Unstable Steering / Excessive Decision Chatter
        if pairwise.delta_steering_reversals >= self.config.steering_reversal_threshold or pairwise.delta_decision_changes >= self.config.decision_instability_threshold:
            failures.append(
                AblationFailureCase(
                    scenario_id=sc_id,
                    scenario_family=fam,
                    run_id=run_id,
                    variant_id=var_id,
                    failure_type="UNSTABLE_STEERING",
                    description=f"Variant {var_id} exhibited high decision instability (+{pairwise.delta_decision_changes} changes, +{pairwise.delta_steering_reversals} reversals vs FULL_PRISM).",
                    timestamp_s=None,
                    details={
                        "full_changes": full_res.decision_changes,
                        "variant_changes": variant_res.decision_changes,
                        "full_reversals": full_res.steering_reversals,
                        "variant_reversals": variant_res.steering_reversals,
                    },
                )
            )

        # 5. Clearance Safety Boundary Violation
        if variant_res.min_clearance_m < self.config.min_safe_clearance_m and full_res.min_clearance_m >= self.config.min_safe_clearance_m:
            failures.append(
                AblationFailureCase(
                    scenario_id=sc_id,
                    scenario_family=fam,
                    run_id=run_id,
                    variant_id=var_id,
                    failure_type="CLEARANCE_VIOLATION",
                    description=f"Variant {var_id} breached the minimum safe clearance ({variant_res.min_clearance_m}m < {self.config.min_safe_clearance_m}m).",
                    timestamp_s=None,
                    details={
                        "variant_min_clearance_m": variant_res.min_clearance_m,
                        "full_min_clearance_m": full_res.min_clearance_m,
                    },
                )
            )

        return failures
