"""Unit tests for RiskAssessment domain type."""

from dataclasses import FrozenInstanceError
import unittest

from app.core.enums import RiskLevel
from app.core.types import RiskAssessment


class TestRiskAssessment(unittest.TestCase):
    """Test suite for RiskAssessment."""

    def test_valid_construction_minimal(self) -> None:
        """Test minimal valid RiskAssessment with default metrics."""
        risk = RiskAssessment(
            object_id=1,
            risk_level=RiskLevel.SAFE,
            risk_score=5.0,
        )
        self.assertEqual(risk.object_id, 1)
        self.assertEqual(risk.risk_level, RiskLevel.SAFE)
        self.assertEqual(risk.risk_score, 5.0)
        self.assertIsNone(risk.ttc)
        self.assertIsNone(risk.cpa_distance)
        self.assertEqual(risk.confidence, 1.0)
        self.assertEqual(risk.timestamp, 0.0)

    def test_valid_construction_full(self) -> None:
        """Test fully populated RiskAssessment with TTC and CPA distance."""
        risk = RiskAssessment(
            object_id=42,
            risk_level=RiskLevel.CRITICAL,
            risk_score=95.5,
            ttc=1.2,
            cpa_distance=0.35,
            confidence=0.9,
            timestamp=55.4,
        )
        self.assertEqual(risk.object_id, 42)
        self.assertEqual(risk.risk_level, RiskLevel.CRITICAL)
        self.assertEqual(risk.risk_score, 95.5)
        self.assertEqual(risk.ttc, 1.2)
        self.assertEqual(risk.cpa_distance, 0.35)
        self.assertEqual(risk.confidence, 0.9)
        self.assertEqual(risk.timestamp, 55.4)

    def test_risk_score_boundaries(self) -> None:
        """Test valid boundary values [0.0, 100.0] for risk score."""
        r0 = RiskAssessment(object_id=1, risk_level=RiskLevel.SAFE, risk_score=0.0)
        r100 = RiskAssessment(object_id=1, risk_level=RiskLevel.CRITICAL, risk_score=100.0)
        self.assertEqual(r0.risk_score, 0.0)
        self.assertEqual(r100.risk_score, 100.0)

    def test_invalid_risk_score_raises(self) -> None:
        """Test that risk score outside [0, 100] raises ValueError."""
        with self.assertRaises(ValueError):
            RiskAssessment(object_id=1, risk_level=RiskLevel.SAFE, risk_score=-0.1)

        with self.assertRaises(ValueError):
            RiskAssessment(object_id=1, risk_level=RiskLevel.CRITICAL, risk_score=100.1)

        with self.assertRaises(ValueError):
            RiskAssessment(object_id=1, risk_level=RiskLevel.SAFE, risk_score=float("nan"))

    def test_invalid_cpa_distance_raises(self) -> None:
        """Test that negative CPA distance raises ValueError."""
        with self.assertRaises(ValueError):
            RiskAssessment(
                object_id=1,
                risk_level=RiskLevel.HIGH,
                risk_score=80.0,
                cpa_distance=-1.0,
            )

    def test_invalid_types_raise(self) -> None:
        """Test that invalid risk_level type raises TypeError."""
        with self.assertRaises(TypeError):
            RiskAssessment(
                object_id=1,
                risk_level="CRITICAL",  # type: ignore[arg-type]
                risk_score=90.0,
            )

    def test_immutability(self) -> None:
        """Test that RiskAssessment is immutable."""
        risk = RiskAssessment(object_id=1, risk_level=RiskLevel.SAFE, risk_score=10.0)
        with self.assertRaises(FrozenInstanceError):
            risk.risk_score = 50.0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
