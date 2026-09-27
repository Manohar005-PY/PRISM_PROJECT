"""CLI entry point for generating PRISM Part 16 Synthetic Experimental Datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.experiments.runner import BatchExperimentRunner
from simulation.experiments.scenarios.registry import ScenarioRegistry
from simulation.experiments.validator import DatasetValidator, DatasetValidationError


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate reproducible synthetic experimental datasets for PRISM Part 16."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--milestone-1",
        action="store_true",
        help="Generate Milestone 1 validation dataset: 15 runs across S01, S02, S03 (5 runs each).",
    )
    group.add_argument(
        "--milestone-2",
        action="store_true",
        help="Generate Milestone 2 initial benchmark dataset: 40+ runs across all 8 scenario families.",
    )
    group.add_argument(
        "--family",
        type=str,
        choices=ScenarioRegistry.get_supported_families(),
        help="Generate a specific scenario family.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs per scenario family (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/datasets/synthetic",
        help="Output directory for generated datasets (default: experiments/datasets/synthetic).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run rigorous dataset validation after generation (default: True).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Base random seed for determinism (default: 1000).",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print(" PRISM Part 16: Synthetic Experimental Dataset Generator")
    print("=" * 70)

    # 1. Select Scenarios
    if args.milestone_1:
        print(f"Generating Milestone 1 Validation Dataset (S01, S02, S03 - {args.runs * 3} runs)...")
        scenarios = ScenarioRegistry.generate_milestone_1_scenarios(
            runs_per_family=args.runs, base_seed=args.seed
        )
        dataset_name = "prism_synthetic_m1_validation"
    elif args.milestone_2 or (not args.milestone_1 and not args.family):
        print(f"Generating Milestone 2 Benchmark Dataset (All 8 Families - {args.runs * 8} runs)...")
        scenarios = ScenarioRegistry.generate_milestone_2_benchmark_scenarios(
            runs_per_family=args.runs, base_seed=args.seed
        )
        dataset_name = "prism_synthetic_benchmark_v1"
    else:
        print(f"Generating Scenario Family: {args.family} ({args.runs} runs)...")
        all_scenarios = ScenarioRegistry.generate_milestone_2_benchmark_scenarios(
            runs_per_family=args.runs, base_seed=args.seed
        )
        scenarios = [s for s in all_scenarios if s.scenario_family == args.family]
        dataset_name = f"prism_synthetic_{args.family.lower()}"

    print(f"Total Scenarios to Execute: {len(scenarios)}")
    print(f"Output Directory: {output_dir.resolve()}")
    print("-" * 70)

    # 2. Execute Batch Generation
    batch_runner = BatchExperimentRunner(base_output_dir=output_dir)

    def on_progress(current: int, total: int, sc_id: str) -> None:
        pct = (current / total) * 100.0
        print(f"  [{current:2d}/{total:2d}] ({pct:5.1f}%) Executing: {sc_id}")

    t_start = sys.modules["time"].perf_counter()
    manifest = batch_runner.run_scenarios(
        scenarios=scenarios,
        dataset_name=dataset_name,
        progress_callback=on_progress,
    )
    t_elapsed = sys.modules["time"].perf_counter() - t_start

    print("-" * 70)
    print("Generation Completed Successfully!")
    print(f"  Total Runs:   {manifest.total_runs}")
    print(f"  Total Frames: {manifest.total_frames}")
    print(f"  Elapsed Time: {t_elapsed:.2f}s ({(manifest.total_frames / max(0.001, t_elapsed)):.1f} frames/sec)")
    print(f"  Manifest:     {output_dir / 'dataset_manifest.json'}")

    # 3. Validate Dataset
    if args.validate:
        print("-" * 70)
        print("Running Dataset Integrity & Schema Validation...")
        validator = DatasetValidator(output_dir)
        try:
            val_res = validator.validate_dataset()
            print(f"  Validation Status: [{val_res['status']}]")
            print(f"  Validated Runs:    {val_res['total_runs']}")
            print(f"  Validated Frames:  {val_res['total_frames']}")
            print(f"  Families Verified: {', '.join(val_res['scenario_families'])}")
            print("=" * 70)
        except DatasetValidationError as e:
            print(f"  VALIDATION FAILED: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
