"""Execution engine running both PRISM and Baseline across dataset runs."""

from __future__ import annotations

from typing import Any, Iterator, Sequence

from simulation.baselines.models import ReactiveBaselineConfig, ReactiveBaselineDecision
from simulation.baselines.reactive_distance import ReactiveDistanceBaseline
from simulation.evaluation.loader import DatasetLoader


class ComparisonRunner:
    """Executes the reactive baseline against frozen Part 16 observation streams with strict isolation."""

    def __init__(self, baseline_config: ReactiveBaselineConfig | None = None) -> None:
        self.baseline_config: ReactiveBaselineConfig = (
            baseline_config if baseline_config is not None else ReactiveBaselineConfig()
        )
        self.baseline = ReactiveDistanceBaseline(self.baseline_config)

    def run_baseline_on_stream(
        self,
        obs_records: Sequence[dict[str, Any]],
        gt_records: Sequence[dict[str, Any]] | None = None,
    ) -> list[ReactiveBaselineDecision]:
        """Execute baseline step-by-step over observation records for a single run."""
        self.baseline.reset()
        decisions: list[ReactiveBaselineDecision] = []

        for i, obs_rec in enumerate(obs_records):
            t = obs_rec["timestamp"]
            f_idx = obs_rec["frame_index"]
            items = obs_rec.get("observations", [])

            ego_pos = (0.0, 0.0)
            if gt_records is not None and i < len(gt_records):
                ego = gt_records[i].get("ego", {})
                ego_pos = (float(ego.get("x", 0.0)), float(ego.get("z", 0.0)))

            dec = self.baseline.step(timestamp=t, frame_index=f_idx, observations=items, ego_position=ego_pos)
            decisions.append(dec)

        return decisions
