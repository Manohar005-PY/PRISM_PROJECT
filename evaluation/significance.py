"""Statistical significance testing: Paired trial differences and confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from evaluation.baselines import SystemEvaluator
from evaluation.metrics import compute_hazard_detection_rate
from evaluation.runner import EvaluationRunner
from simulation.scenarios import Scenario


@dataclass(frozen=True)
class PairedSignificanceResult:
    """Rigorous statistical hypothesis test result on paired experimental trials."""

    metric_name: str
    sample_size: int
    mean_baseline: float
    mean_refined: float
    mean_difference: float
    std_err_diff: float
    ci_95_diff_lower: float
    ci_95_diff_upper: float
    is_statistically_significant: bool


class SignificanceTester:
    """Executes paired hypothesis tests across multi-trial experimental runs."""

    @staticmethod
    def evaluate_paired_difference(
        baseline_evaluator: SystemEvaluator,
        refined_evaluator: SystemEvaluator,
        scenarios: Mapping[str, Scenario],
        seeds: Sequence[int] = (42, 101, 2024, 777, 999),
        metric_extractor=compute_hazard_detection_rate,
        metric_name: str = "hazard_detection_rate",
    ) -> PairedSignificanceResult:
        """Run repeated paired trials across identical random seeds and test difference from 0."""
        b_scores: list[float] = []
        r_scores: list[float] = []
        diffs: list[float] = []

        for seed in seeds:
            runner = EvaluationRunner(random_seed=seed)

            # Run baseline
            b_samples = []
            for sc in scenarios.values():
                b_samples.extend(runner.run_scenario(sc, baseline_evaluator))
            b_val = metric_extractor(b_samples)
            b_scores.append(b_val)

            # Run refined
            r_samples = []
            for sc in scenarios.values():
                r_samples.extend(runner.run_scenario(sc, refined_evaluator))
            r_val = metric_extractor(r_samples)
            r_scores.append(r_val)

            diffs.append(r_val - b_val)

        n = len(diffs)
        mean_b = sum(b_scores) / n
        mean_r = sum(r_scores) / n
        mean_diff = sum(diffs) / n

        if n > 1:
            var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
            std_diff = math.sqrt(var_diff)
            std_err = std_diff / math.sqrt(n)
            margin = 1.96 * std_err
        else:
            std_err = 0.0
            margin = 0.0

        ci_lower = mean_diff - margin
        ci_upper = mean_diff + margin

        # Statistically significant at 95% level if CI on difference does not contain 0.0
        is_sig = (ci_lower > 0.0) or (ci_upper < 0.0)

        return PairedSignificanceResult(
            metric_name=metric_name,
            sample_size=n,
            mean_baseline=mean_b,
            mean_refined=mean_r,
            mean_difference=mean_diff,
            std_err_diff=std_err,
            ci_95_diff_lower=ci_lower,
            ci_95_diff_upper=ci_upper,
            is_statistically_significant=is_sig,
        )
