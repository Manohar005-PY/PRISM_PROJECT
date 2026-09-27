"""Automated failure mode and diagnostic discrepancy analyzer for real-world testing."""

from __future__ import annotations

from typing import Sequence

from simulation.real_world.models import (
    RealWorldConfig,
    RealWorldFailureCase,
    RealWorldRunResult,
)


class RealWorldFailureAnalyzer:
    """Classifies real-world diagnostic discrepancies across perception, tracking, and decisions."""

    def __init__(self, config: RealWorldConfig | None = None) -> None:
        self.config: RealWorldConfig = config if config is not None else RealWorldConfig()

    def analyze_run(self, result: RealWorldRunResult) -> list[RealWorldFailureCase]:
        """Detect and classify any real-world failure modes in a single video run."""
        failures: list[RealWorldFailureCase] = []
        r_id = result.run_id
        fam = result.scenario_family

        # 1. Missed Warning on Collision Path
        if result.has_collision and not result.warned:
            failures.append(
                RealWorldFailureCase(
                    run_id=r_id,
                    scenario_family=fam,
                    failure_type="MISSED_WARNING",
                    description=f"PRISM failed to trigger any warning alert before collision at t={result.collision_timestamp_s}s.",
                    timestamp_s=result.collision_timestamp_s,
                    details={"collision_timestamp_s": result.collision_timestamp_s},
                )
            )

        # 2. Late Warning on Collision Path (Lead Time < 1.0s)
        if result.has_collision and result.warning_lead_time_s is not None:
            if result.warning_lead_time_s < 1.0:
                failures.append(
                    RealWorldFailureCase(
                        run_id=r_id,
                        scenario_family=fam,
                        failure_type="LATE_WARNING",
                        description=f"Warning lead time ({result.warning_lead_time_s:.2f}s) was critically low (< 1.0s) before closest approach.",
                        timestamp_s=result.first_warning_timestamp_s,
                        details={"lead_time_s": result.warning_lead_time_s},
                    )
                )

        # 3. False Warning in Safe Scenario
        if not result.has_collision and result.false_warning:
            failures.append(
                RealWorldFailureCase(
                    run_id=r_id,
                    scenario_family=fam,
                    failure_type="FALSE_WARNING",
                    description=f"False warning alert triggered at t={result.first_warning_timestamp_s:.2f}s in a non-colliding scenario.",
                    timestamp_s=result.first_warning_timestamp_s,
                    details={"first_warning_s": result.first_warning_timestamp_s},
                )
            )

        # 4. Track Fragmentation
        if result.tracking_continuity_ratio < 0.60:
            failures.append(
                RealWorldFailureCase(
                    run_id=r_id,
                    scenario_family=fam,
                    failure_type="TRACK_FRAGMENTATION",
                    description=f"Low tracking continuity ratio ({result.tracking_continuity_ratio * 100:.1f}%) indicates track dropouts or ID switches.",
                    timestamp_s=None,
                    details={"continuity_ratio": result.tracking_continuity_ratio},
                )
            )

        # 5. Steering Flutter
        if result.steering_reversals >= 3:
            failures.append(
                RealWorldFailureCase(
                    run_id=r_id,
                    scenario_family=fam,
                    failure_type="STEERING_FLUTTER",
                    description=f"High steering oscillations ({result.steering_reversals} reversals) detected during maneuver.",
                    timestamp_s=None,
                    details={"steering_reversals": result.steering_reversals},
                )
            )

        return failures
