"""PRISM Part 19: Ablation & Sensitivity Analysis package."""

from simulation.ablation.aggregation import (
    AblationAggregator,
    compute_cohens_d,
    compute_mean,
    compute_median,
    compute_std,
)
from simulation.ablation.analyzer import PRISMAblationAnalyzer
from simulation.ablation.failure_analysis import AblationFailureAnalyzer
from simulation.ablation.metrics import (
    WARNING_ACTIONS,
    WARNING_RISK_LEVELS,
    AblationMetricsEvaluator,
)
from simulation.ablation.models import (
    FROZEN_SENSITIVITY_SPECS,
    AblationConfig,
    AblationFailureCase,
    AblationVariant,
    PairwiseVariantComparison,
    SensitivityParamSpec,
    SensitivitySettingResult,
    VariantFrameOutput,
    VariantRunResult,
)
from simulation.ablation.runner import AblationRunner
from simulation.ablation.sensitivity import SensitivityAnalyzer
from simulation.ablation.variants import create_variant_pipeline

__all__ = [
    "AblationAggregator",
    "AblationConfig",
    "AblationFailureAnalyzer",
    "AblationFailureCase",
    "AblationMetricsEvaluator",
    "AblationRunner",
    "AblationVariant",
    "FROZEN_SENSITIVITY_SPECS",
    "PRISMAblationAnalyzer",
    "PairwiseVariantComparison",
    "SensitivityAnalyzer",
    "SensitivityParamSpec",
    "SensitivitySettingResult",
    "VariantFrameOutput",
    "VariantRunResult",
    "WARNING_ACTIONS",
    "WARNING_RISK_LEVELS",
    "compute_cohens_d",
    "compute_mean",
    "compute_median",
    "compute_std",
    "create_variant_pipeline",
]
