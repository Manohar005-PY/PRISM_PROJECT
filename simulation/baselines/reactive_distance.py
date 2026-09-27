"""Pure reactive distance-based obstacle avoidance baseline."""

from __future__ import annotations

import math
from typing import Any, Sequence

from simulation.baselines.models import ReactiveBaselineConfig, ReactiveBaselineDecision


class ReactiveDistanceBaseline:
    """Conventional reactive baseline mapping current raw distance observations to navigation decisions.

    Guarantees:
    - Never calculates velocity, trajectory extrapolation, TTC, dCPA, or risk scores.
    - Operates strictly on raw observation coordinates (x, z) at current timestep t.
    - Fully deterministic decision and heading rules.
    - Zero state leakage across scenario runs via explicit reset().
    """

    def __init__(self, config: ReactiveBaselineConfig | None = None) -> None:
        self.config: ReactiveBaselineConfig = config if config is not None else ReactiveBaselineConfig()
        self._last_action: str = "CONTINUE"
        self._action_hold_counter: int = 0

    def reset(self) -> None:
        """Reset internal hysteresis and state counters between independent runs."""
        self._last_action = "CONTINUE"
        self._action_hold_counter = 0

    def step(
        self,
        timestamp: float,
        frame_index: int,
        observations: Sequence[dict[str, Any] | Any],
        ego_position: tuple[float, float] = (0.0, 0.0),
    ) -> ReactiveBaselineDecision:
        """Process instantaneous observations at timestep t relative to user position."""
        ego_x, ego_z = ego_position
        valid_obs: list[tuple[float, float, int | str | None, float]] = []

        for obs in observations:
            # Support both dict and object interfaces
            if isinstance(obs, dict):
                is_valid = obs.get("valid", True)
                if not is_valid:
                    continue
                obs_x = float(obs["x"])
                obs_z = float(obs["z"])
                obj_id = obs.get("object_id")
            else:
                is_valid = getattr(obs, "valid", True)
                if not is_valid:
                    continue
                obs_x = float(obs.x)
                obs_z = float(obs.z)
                obj_id = getattr(obs, "object_id", None)

            # User-relative coordinates
            rel_x = obs_x - ego_x
            rel_z = obs_z - ego_z
            dist = math.sqrt(rel_x * rel_x + rel_z * rel_z)
            if math.isfinite(dist):
                valid_obs.append((dist, rel_x, obj_id, rel_z))

        if not valid_obs:
            # No visible obstacles in current field of view
            raw_action = "CONTINUE"
            warning_lvl = "SAFE"
            hdg = 0.0
            closest_id = None
            min_dist = None
            is_warn = False
        else:
            # Sort by Euclidean distance to find closest instantaneous obstacle
            valid_obs.sort(key=lambda item: item[0])
            min_dist, closest_x, closest_id, closest_z = valid_obs[0]

            # Deterministic Threshold Hierarchy
            if min_dist > self.config.warning_distance_m:
                raw_action = "CONTINUE"
                warning_lvl = "SAFE"
                hdg = 0.0
                is_warn = False
            elif min_dist > self.config.avoidance_distance_m:
                raw_action = "SLOW_DOWN"
                warning_lvl = "HIGH"
                hdg = 0.0
                is_warn = True
            elif min_dist > self.config.stop_distance_m:
                # Emergency lateral steering away from closest obstacle
                if closest_x >= 0.0:
                    raw_action = "TURN_LEFT"
                    hdg = self.config.lateral_turn_angle_rad
                else:
                    raw_action = "TURN_RIGHT"
                    hdg = -self.config.lateral_turn_angle_rad
                warning_lvl = "CRITICAL"
                is_warn = True
            else:
                raw_action = "STOP"
                warning_lvl = "CRITICAL"
                hdg = 0.0
                is_warn = True

        # Apply Hysteresis to prevent 1-frame flickering
        # Higher-severity actions (STOP > TURN > SLOW_DOWN > CONTINUE) trigger immediately
        action_severity = {"CONTINUE": 0, "SLOW_DOWN": 1, "TURN_LEFT": 2, "TURN_RIGHT": 2, "STOP": 3}
        current_sev = action_severity.get(raw_action, 0)
        prev_sev = action_severity.get(self._last_action, 0)

        if current_sev >= prev_sev or self._action_hold_counter >= self.config.hysteresis_frames:
            final_action = raw_action
            if final_action != self._last_action:
                self._last_action = final_action
                self._action_hold_counter = 1
            else:
                self._action_hold_counter += 1
        else:
            final_action = self._last_action
            self._action_hold_counter += 1

        # Re-derive heading if action was held by hysteresis
        if final_action == "TURN_LEFT":
            hdg = self.config.lateral_turn_angle_rad
            warning_lvl = "CRITICAL"
            is_warn = True
        elif final_action == "TURN_RIGHT":
            hdg = -self.config.lateral_turn_angle_rad
            warning_lvl = "CRITICAL"
            is_warn = True
        elif final_action == "STOP":
            hdg = 0.0
            warning_lvl = "CRITICAL"
            is_warn = True
        elif final_action == "SLOW_DOWN":
            hdg = 0.0
            warning_lvl = "HIGH"
            is_warn = True
        elif final_action == "CONTINUE":
            hdg = 0.0
            warning_lvl = "SAFE"
            is_warn = False

        return ReactiveBaselineDecision(
            timestamp=timestamp,
            frame_index=frame_index,
            action=final_action,
            warning_level=warning_lvl,
            recommended_heading_rad=hdg,
            closest_obstacle_id=closest_id,
            min_observed_distance_m=round(min_dist, 4) if min_dist is not None else None,
            is_warning=is_warn,
        )
