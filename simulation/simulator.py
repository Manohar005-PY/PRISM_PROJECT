"""Synthetic simulation engine and observation generator for PRISM.

The simulator steps through predefined or custom scenarios, propagating true physical
kinematics via ConstantVelocityModel and producing ground-truth history records alongside
controlled, noisy synthetic observations.

CRITICAL ARCHITECTURAL NOTE:
The SyntheticObservationGenerator is a synthetic testing adapter designed strictly to produce
valid Part 1 ObjectObservation domain instances. It is NOT a real camera projection or homography
model (camera calibration and perspective projection belong to Part 3).
"""

from __future__ import annotations

import math

from app.core.types import BoundingBox, ObjectObservation, Vector2
from simulation.entities import EgoEntity, SimulatedEntity
from simulation.ground_truth import GroundTruthFrame, GroundTruthSequence
from simulation.motion import ConstantVelocityModel
from simulation.noise import GaussianNoiseModel
from simulation.scenarios import Scenario


class SyntheticObservationGenerator:
    """Generates synthetic ObjectObservation instances from ground-plane coordinates.

    NOTE: This is a synthetic observation model and is NOT a camera projection model.
    It produces canonical image-space BoundingBoxes to fulfill the Part 1 ObjectObservation
    contract without leaking camera intrinsics into the simulation foundation.
    """

    def __init__(
        self,
        default_confidence: float = 0.95,
        synthetic_scale: float = 50.0,
        synthetic_origin_x: float = 320.0,
        synthetic_origin_y: float = 240.0,
        box_width: float = 40.0,
        box_height: float = 80.0,
    ) -> None:
        self.default_confidence: float = default_confidence
        self.synthetic_scale: float = synthetic_scale
        self.synthetic_origin_x: float = synthetic_origin_x
        self.synthetic_origin_y: float = synthetic_origin_y
        self.box_width: float = box_width
        self.box_height: float = box_height

    def generate_observation(
        self,
        entity: SimulatedEntity,
        observed_position: Vector2,
        confidence: float | None = None,
    ) -> ObjectObservation:
        """Convert a ground-plane position into a synthetic ObjectObservation.

        Args:
            entity: True simulated entity (provides identity, class, timestamp).
            observed_position: Measured position (may include injected noise).
            confidence: Observation confidence score in [0.0, 1.0].

        Returns:
            Valid ObjectObservation instance.
        """
        conf = self.default_confidence if confidence is None else confidence

        # Map ground coordinates to synthetic image coordinates
        cx = self.synthetic_origin_x + observed_position.x * self.synthetic_scale
        bottom_y = self.synthetic_origin_y + observed_position.z * self.synthetic_scale

        half_w = self.box_width / 2.0
        x_min = cx - half_w
        x_max = cx + half_w
        y_min = bottom_y - self.box_height
        y_max = bottom_y

        bbox = BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )

        return ObjectObservation(
            object_id=entity.object_id,
            class_id=entity.class_id,
            confidence=conf,
            bbox=bbox,
            timestamp=entity.timestamp,
        )


class Simulator:
    """Deterministic simulation engine for PRISM scenarios.

    Executes a scenario over discrete time steps, recording ground truth and
    generating synthetic noisy observations.
    """

    def __init__(
        self,
        scenario: Scenario,
        noise_model: GaussianNoiseModel | None = None,
        observation_generator: SyntheticObservationGenerator | None = None,
    ) -> None:
        if not isinstance(scenario, Scenario):
            raise TypeError(f"scenario must be a Scenario, got {type(scenario)}")
        if noise_model is not None and not isinstance(noise_model, GaussianNoiseModel):
            raise TypeError(f"noise_model must be a GaussianNoiseModel or None, got {type(noise_model)}")
        if observation_generator is not None and not isinstance(observation_generator, SyntheticObservationGenerator):
            raise TypeError(f"observation_generator must be a SyntheticObservationGenerator or None, got {type(observation_generator)}")

        self.scenario: Scenario = scenario
        self.noise_model: GaussianNoiseModel | None = noise_model
        self.obs_gen: SyntheticObservationGenerator = (
            observation_generator if observation_generator is not None else SyntheticObservationGenerator()
        )

    def run(self) -> tuple[GroundTruthSequence, list[list[ObjectObservation]]]:
        """Execute the simulation scenario from t=0.0 to t=duration.

        Returns:
            A tuple of (GroundTruthSequence, list[list[ObjectObservation]])
            where each element in the observation list corresponds to a simulation frame.
        """
        frames: list[GroundTruthFrame] = []
        observation_history: list[list[ObjectObservation]] = []

        dt = self.scenario.timestep
        duration = self.scenario.duration
        num_steps = int(round(duration / dt)) + 1

        init_ego = self.scenario.ego_initial
        init_objects = list(self.scenario.objects_initial)

        for step in range(num_steps):
            current_time = round(step * dt, 8)

            # Exact kinematic propagation from initial state: p(t) = p0 + v * t
            ego_pos = ConstantVelocityModel.advance(init_ego.position, init_ego.velocity, current_time)
            frame_ego = EgoEntity(
                position=ego_pos,
                velocity=init_ego.velocity,
                timestamp=current_time,
            )

            frame_objects: list[SimulatedEntity] = []
            frame_observations: list[ObjectObservation] = []

            for obj in init_objects:
                obj_pos = ConstantVelocityModel.advance(obj.position, obj.velocity, current_time)
                entity = SimulatedEntity(
                    object_id=obj.object_id,
                    class_id=obj.class_id,
                    position=obj_pos,
                    velocity=obj.velocity,
                    timestamp=current_time,
                )
                frame_objects.append(entity)

                # Generate noisy observation (pure ground truth is never mutated)
                if self.noise_model is not None:
                    observed_pos = self.noise_model.apply_position_noise(entity.position)
                else:
                    observed_pos = entity.position

                obs = self.obs_gen.generate_observation(
                    entity=entity,
                    observed_position=observed_pos,
                )
                frame_observations.append(obs)

            frames.append(
                GroundTruthFrame(
                    timestamp=current_time,
                    ego=frame_ego,
                    objects=tuple(frame_objects),
                )
            )
            observation_history.append(frame_observations)

        return GroundTruthSequence(frames=tuple(frames)), observation_history
