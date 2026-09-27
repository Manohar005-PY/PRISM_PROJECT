"""Unit tests for ExperimentProtocol and configuration validation."""

import unittest

from evaluation.protocol import ExperimentProtocol, LatencyConfig, NoiseConfig


class TestExperimentProtocol(unittest.TestCase):
    """Test suite for ExperimentProtocol."""

    def test_protocol_creation_and_serialization(self) -> None:
        proto = ExperimentProtocol(
            experiment_id="EXP_BENCH_01",
            random_seed=42,
            scenario_ids=("HEAD_ON", "CROSSING"),
            number_of_trials=3,
            timestep=0.1,
            duration=3.0,
            noise_config=NoiseConfig(sigma_x=0.05, sigma_z=0.05, detection_drop_rate=0.05),
            latency_config=LatencyConfig(simulated_latency_ms=20.0),
        )

        self.assertEqual(proto.experiment_id, "EXP_BENCH_01")
        self.assertEqual(proto.number_of_trials, 3)

        d = proto.to_dict()
        self.assertEqual(d["experiment_id"], "EXP_BENCH_01")

        recovered = ExperimentProtocol.from_dict(d)
        self.assertEqual(recovered.experiment_id, proto.experiment_id)
        self.assertEqual(recovered.random_seed, proto.random_seed)
        self.assertEqual(recovered.noise_config.detection_drop_rate, 0.05)


if __name__ == "__main__":
    unittest.main()
