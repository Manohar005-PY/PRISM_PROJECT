"""Central registry and parameter grid generator for PRISM scenario families."""

from __future__ import annotations

import math
from typing import Callable, Iterator, Sequence

from simulation.experiments.models import ScenarioDefinition
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

ALL_SCENARIO_FAMILIES: list[str] = [
    "S01_stationary",
    "S02_head_on",
    "S03_lateral_crossing",
    "S04_moving_away",
    "S05_direction_change",
    "S06_multiple_objects",
    "S07_corridor_squeeze",
    "S08_measurement_noise",
]


class ScenarioRegistry:
    """Discovers and instantiates parameterized scenario suites for Part 16."""

    @classmethod
    def get_supported_families(cls) -> list[str]:
        """Return list of all 8 supported scenario families."""
        return list(ALL_SCENARIO_FAMILIES)

    @classmethod
    def generate_milestone_1_scenarios(cls, runs_per_family: int = 5, base_seed: int = 1000) -> list[ScenarioDefinition]:
        """Generate Milestone 1 validation dataset: S01, S02, S03 with 5 runs each (15 runs total)."""
        scenarios: list[ScenarioDefinition] = []

        # S01: 5 runs varying distance and lateral offset
        s01_params = [
            (4.0, 0.0, 1.0),
            (5.5, 0.0, 1.0),
            (7.0, 0.0, 1.2),
            (5.0, 0.3, 1.0),
            (6.0, -0.4, 0.8),
        ]
        for idx, (dist, offset, spd) in enumerate(s01_params[:runs_per_family], start=1):
            sc = create_s01_stationary_scenario(
                initial_distance=dist,
                lateral_offset=offset,
                user_speed=spd,
                seed=base_seed + idx,
                run_idx=idx,
            )
            scenarios.append(sc)

        # S02: 5 runs varying separation and approach speeds
        s02_params = [
            (5.0, 1.0, 1.0, 0.0),
            (6.5, 1.0, 1.2, 0.0),
            (8.0, 0.8, 1.0, 0.0),
            (6.0, 1.0, 1.0, 0.25),
            (7.0, 1.2, 0.9, -0.25),
        ]
        for idx, (sep, u_spd, p_spd, offset) in enumerate(s02_params[:runs_per_family], start=1):
            sc = create_s02_head_on_scenario(
                initial_separation=sep,
                user_speed=u_spd,
                person_speed=p_spd,
                lateral_offset=offset,
                seed=base_seed + 100 + idx,
                run_idx=idx,
            )
            scenarios.append(sc)

        # S03: 5 runs (3 collision, 2 miss)
        s03_params = [
            ("coll", 3.5, 1.0, 1.0, -3.5, 0.0),
            ("coll", 4.5, 1.2, 1.0, -4.0, 0.0),
            ("coll", 3.0, 0.8, 0.9, -3.0, 0.0),
            ("miss", 3.5, 1.0, 1.0, -3.5, 1.5),
            ("miss", 4.0, 1.0, 1.0, -4.0, 2.0),
        ]
        for idx, (ctype, z, u_spd, p_spd, start_x, toff) in enumerate(s03_params[:runs_per_family], start=1):
            if ctype == "coll":
                sc = create_s03_crossing_collision_scenario(
                    crossing_z=z,
                    user_speed=u_spd,
                    pedestrian_speed=p_spd,
                    start_x=start_x,
                    seed=base_seed + 200 + idx,
                    run_idx=idx,
                )
            else:
                sc = create_s03_crossing_miss_scenario(
                    crossing_z=z,
                    user_speed=u_spd,
                    time_offset=toff,
                    start_x=start_x,
                    seed=base_seed + 200 + idx,
                    run_idx=idx,
                )
            scenarios.append(sc)

        return scenarios

    @classmethod
    def generate_milestone_2_benchmark_scenarios(
        cls,
        runs_per_family: int = 5,
        base_seed: int = 1000,
    ) -> list[ScenarioDefinition]:
        """Generate Milestone 2 initial benchmark dataset across all 8 scenario families (40+ runs)."""
        scenarios: list[ScenarioDefinition] = []

        # 1. S01: Stationary
        for idx in range(1, runs_per_family + 1):
            dist = 4.0 + (idx - 1) * 0.8
            offset = 0.15 * (idx - 3)
            spd = 0.8 + (idx - 1) * 0.15
            scenarios.append(
                create_s01_stationary_scenario(
                    initial_distance=dist,
                    lateral_offset=offset,
                    user_speed=spd,
                    seed=base_seed + idx,
                    run_idx=idx,
                )
            )

        # 2. S02: Head-On
        for idx in range(1, runs_per_family + 1):
            sep = 4.5 + (idx - 1) * 0.9
            u_spd = 0.9 + (idx % 2) * 0.3
            p_spd = 0.9 + ((idx + 1) % 2) * 0.3
            scenarios.append(
                create_s02_head_on_scenario(
                    initial_separation=sep,
                    user_speed=u_spd,
                    person_speed=p_spd,
                    seed=base_seed + 100 + idx,
                    run_idx=idx,
                )
            )

        # 3. S03: Lateral Crossing (mix of collision and miss)
        for idx in range(1, runs_per_family + 1):
            if idx <= 3:
                scenarios.append(
                    create_s03_crossing_collision_scenario(
                        crossing_z=3.0 + idx * 0.5,
                        user_speed=1.0,
                        start_x=-3.0 - idx * 0.3,
                        seed=base_seed + 200 + idx,
                        run_idx=idx,
                    )
                )
            else:
                scenarios.append(
                    create_s03_crossing_miss_scenario(
                        crossing_z=3.5,
                        user_speed=1.0,
                        time_offset=1.2 + (idx - 3) * 0.6,
                        seed=base_seed + 200 + idx,
                        run_idx=idx,
                    )
                )

        # 4. S04: Moving Away
        for idx in range(1, runs_per_family + 1):
            init_dist = 2.5 + (idx - 1) * 0.8
            t_spd = 1.3 + (idx - 1) * 0.2
            scenarios.append(
                create_s04_moving_away_scenario(
                    initial_distance=init_dist,
                    user_speed=1.0,
                    target_speed=t_spd,
                    seed=base_seed + 300 + idx,
                    run_idx=idx,
                )
            )

        # 5. S05: Direction Change
        angles = [math.pi / 4.0, math.pi / 2.0, 3 * math.pi / 4.0, -math.pi / 2.0, math.pi / 3.0]
        for idx in range(1, runs_per_family + 1):
            angle = angles[(idx - 1) % len(angles)]
            t_turn = 2.0 + (idx - 1) * 0.4
            scenarios.append(
                create_s05_direction_change_scenario(
                    turn_time=t_turn,
                    turn_angle_rad=angle,
                    user_speed=1.0,
                    seed=base_seed + 400 + idx,
                    run_idx=idx,
                )
            )

        # 6. S06: Multiple Objects
        obj_counts = [2, 3, 3, 5, 5]
        for idx in range(1, runs_per_family + 1):
            n_obj = obj_counts[(idx - 1) % len(obj_counts)]
            scenarios.append(
                create_s06_multiple_objects_scenario(
                    num_objects=n_obj,
                    user_speed=1.0,
                    seed=base_seed + 500 + idx,
                    run_idx=idx,
                )
            )

        # 7. S07: Corridor Squeeze
        widths = [2.0, 2.4, 2.8, 2.2, 2.6]
        obs_offsets = [0.35, 0.45, -0.50, 0.30, -0.40]
        for idx in range(1, runs_per_family + 1):
            w = widths[(idx - 1) % len(widths)]
            ox = obs_offsets[(idx - 1) % len(obs_offsets)]
            scenarios.append(
                create_s07_corridor_squeeze_scenario(
                    corridor_width=w,
                    obstacle_x=ox,
                    seed=base_seed + 600 + idx,
                    run_idx=idx,
                )
            )

        # 8. S08: Measurement Noise
        noise_levels = ["NO_NOISE", "LOW_NOISE", "MEDIUM_NOISE", "HIGH_NOISE", "DROPOUT_ONLY"]
        for idx in range(1, runs_per_family + 1):
            n_lvl = noise_levels[(idx - 1) % len(noise_levels)]
            scenarios.append(
                create_s08_noise_scenario(
                    noise_level=n_lvl,
                    seed=base_seed + 700 + idx,
                    run_idx=idx,
                )
            )

        return scenarios
