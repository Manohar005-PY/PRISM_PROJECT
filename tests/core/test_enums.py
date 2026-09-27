"""Unit tests for PRISM enums."""

import unittest

from app.core.enums import DecisionType, RiskLevel


class TestEnums(unittest.TestCase):
    """Test suite for semantic domain enums."""

    def test_risk_level_values(self) -> None:
        """Verify RiskLevel enum members and string representations."""
        expected_members = {"SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
        actual_members = {member.value for member in RiskLevel}
        self.assertEqual(expected_members, actual_members)
        self.assertEqual(RiskLevel.SAFE.value, "SAFE")
        self.assertEqual(RiskLevel.LOW.value, "LOW")
        self.assertEqual(RiskLevel.MEDIUM.value, "MEDIUM")
        self.assertEqual(RiskLevel.HIGH.value, "HIGH")
        self.assertEqual(RiskLevel.CRITICAL.value, "CRITICAL")

    def test_decision_type_values(self) -> None:
        """Verify DecisionType enum members and string representations."""
        expected_members = {"NO_ACTION", "CAUTION", "WARNING", "STOP"}
        actual_members = {member.value for member in DecisionType}
        self.assertEqual(expected_members, actual_members)
        self.assertEqual(DecisionType.NO_ACTION.value, "NO_ACTION")
        self.assertEqual(DecisionType.CAUTION.value, "CAUTION")
        self.assertEqual(DecisionType.WARNING.value, "WARNING")
        self.assertEqual(DecisionType.STOP.value, "STOP")


if __name__ == "__main__":
    unittest.main()
