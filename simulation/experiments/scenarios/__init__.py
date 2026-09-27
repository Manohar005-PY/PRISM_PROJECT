"""Scenario families for PRISM Part 16 Synthetic Experimental Benchmark."""

from simulation.experiments.scenarios.s01_stationary import create_s01_stationary_scenario
from simulation.experiments.scenarios.s02_head_on import create_s02_head_on_scenario
from simulation.experiments.scenarios.s03_lateral_crossing import (
    create_s03_crossing_collision_scenario,
    create_s03_crossing_miss_scenario,
)
from simulation.experiments.scenarios.s04_moving_away import create_s04_moving_away_scenario
from simulation.experiments.scenarios.s05_direction_change import create_s05_direction_change_scenario
from simulation.experiments.scenarios.s06_multiple_objects import create_s06_multiple_objects_scenario
from simulation.experiments.scenarios.s07_corridor_squeeze import create_s07_corridor_squeeze_scenario
from simulation.experiments.scenarios.s08_measurement_noise import create_s08_noise_scenario
from simulation.experiments.scenarios.registry import ScenarioRegistry

__all__ = [
    "create_s01_stationary_scenario",
    "create_s02_head_on_scenario",
    "create_s03_crossing_collision_scenario",
    "create_s03_crossing_miss_scenario",
    "create_s04_moving_away_scenario",
    "create_s05_direction_change_scenario",
    "create_s06_multiple_objects_scenario",
    "create_s07_corridor_squeeze_scenario",
    "create_s08_noise_scenario",
    "ScenarioRegistry",
]
