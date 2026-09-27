"""PRISM Evaluation Subsystem - Scientific validation, failure analysis, ablation, and benchmarking."""

from evaluation.ablation import AblationMetrics, ControlledAblationStudy
from evaluation.aggregation import StatisticalSummary, aggregate_values
from evaluation.baselines import (
    DistanceBaseline,
    PRISMEvaluator,
    SystemEvaluator,
    TTCBaseline,
)
from evaluation.calibration import DepthBinMetrics, PerceptionCalibrator
from evaluation.comparison import MetricComparison, SystemComparator
from evaluation.diagnosis import CausalFailureDiagnoser, DiagnosisRecord
from evaluation.error_analysis import MultiObjectiveErrorAnalyzer, ParetoTradeoffProfile
from evaluation.experiment import ExperimentRun, MetricResult, TrialResult
from evaluation.ground_truth import GroundTruthProvider, SyntheticGroundTruthProvider
from evaluation.metrics import (
    compute_decision_switch_rate,
    compute_false_alarm_rate,
    compute_hazard_detection_rate,
    compute_position_rmse,
    compute_reaction_time,
    compute_unnecessary_stop_rate,
)
from evaluation.models import (
    DecisionRecord,
    EvaluationSample,
    GroundTruthState,
    PredictionRecord,
    RiskRecord,
)
from evaluation.persistence import export_evaluation_json, export_samples_csv
from evaluation.protocol import ExperimentProtocol, LatencyConfig, NoiseConfig
from evaluation.refinement import create_refined_prism_evaluator
from evaluation.refinement_protocol import (
    RefinementConfig,
    get_development_scenario_partition,
    get_holdout_scenario_partition,
)
from evaluation.reporting import (
    format_baseline_comparison_table,
    format_calibration_table,
    format_robustness_table,
    format_sensitivity_table,
)
from evaluation.robustness import RobustnessAnalyzer, RobustnessResult
from evaluation.runner import EvaluationRunner, create_prism_evaluator_system
from evaluation.scenarios import get_canonical_benchmark_suite
from evaluation.sensitivity import SensitivityAnalyzer, SensitivityResult
from evaluation.significance import PairedSignificanceResult, SignificanceTester
from evaluation.statistics import compute_statistics
from evaluation.validation import FailureCase, FailureCategory, FailureClassifier

__all__ = [
    "AblationMetrics",
    "CausalFailureDiagnoser",
    "ControlledAblationStudy",
    "DecisionRecord",
    "DepthBinMetrics",
    "DiagnosisRecord",
    "DistanceBaseline",
    "EvaluationRunner",
    "EvaluationSample",
    "ExperimentProtocol",
    "ExperimentRun",
    "FailureCase",
    "FailureCategory",
    "FailureClassifier",
    "GroundTruthProvider",
    "GroundTruthState",
    "LatencyConfig",
    "MetricComparison",
    "MetricResult",
    "MultiObjectiveErrorAnalyzer",
    "NoiseConfig",
    "PRISMEvaluator",
    "PairedSignificanceResult",
    "ParetoTradeoffProfile",
    "PerceptionCalibrator",
    "PredictionRecord",
    "RefinementConfig",
    "RiskRecord",
    "RobustnessAnalyzer",
    "RobustnessResult",
    "SensitivityAnalyzer",
    "SensitivityResult",
    "SignificanceTester",
    "StatisticalSummary",
    "SyntheticGroundTruthProvider",
    "SystemComparator",
    "SystemEvaluator",
    "TTCBaseline",
    "TrialResult",
    "aggregate_values",
    "compute_decision_switch_rate",
    "compute_false_alarm_rate",
    "compute_hazard_detection_rate",
    "compute_position_rmse",
    "compute_reaction_time",
    "compute_statistics",
    "compute_unnecessary_stop_rate",
    "create_prism_evaluator_system",
    "create_refined_prism_evaluator",
    "export_evaluation_json",
    "export_samples_csv",
    "format_baseline_comparison_table",
    "format_calibration_table",
    "format_robustness_table",
    "format_sensitivity_table",
    "get_canonical_benchmark_suite",
    "get_development_scenario_partition",
    "get_holdout_scenario_partition",
]
