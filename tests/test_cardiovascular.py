"""Tests for the cardiovascular plugin and its cardiovascular-only workflow."""

import pytest

from Agents.plugins.cardiovascular import CardiovascularPlugin
from Agents.reportanalyzer import validate_parameters

HEALTHY = {
    "age": 35,
    "systolic_bp": 115,
    "diastolic_bp": 72,
    "total_cholesterol": 180,
    "ldl": 90,
    "hdl": 55,
    "smoker": False,
}

AT_RISK = {
    "age": 62,
    "systolic_bp": 168,
    "diastolic_bp": 102,
    "total_cholesterol": 265,
    "ldl": 155,
    "hdl": 34,
    "smoker": True,
}


def test_plugin_declares_its_contract():
    assert CardiovascularPlugin.name == "cardiovascular"
    # The required fields are exactly the ML model's feature set, so a case
    # the plugin accepts can always be scored by the model.
    assert CardiovascularPlugin.required_fields == [
        "age", "systolic_bp", "total_cholesterol",
    ]
    # Smoking status is optional: absent means "not supplied", not "no".
    assert "smoker" in CardiovascularPlugin.optional_fields


def test_rule_risk_is_low_for_a_healthy_profile():
    risk = CardiovascularPlugin().rule_risk(HEALTHY)
    assert risk["risk_level"] in ("Low", "Moderate")
    assert 0.0 <= risk["risk_probability"] <= 1.0
    assert risk["prediction_method"] == "rule_based"


def test_rule_risk_is_high_for_an_at_risk_profile():
    risk = CardiovascularPlugin().rule_risk(AT_RISK)
    assert risk["risk_level"] in ("High", "Critical")
    assert risk["risk_probability"] > 0.4


def test_rule_risk_reports_factors_and_recommendations():
    risk = CardiovascularPlugin().rule_risk(AT_RISK)
    parameters = {factor["parameter"] for factor in risk["risk_factors"]}
    assert "systolic_bp" in parameters
    assert "total_cholesterol" in parameters
    assert "smoker" in parameters
    assert risk["recommendations"]


def test_predict_risk_falls_back_to_rules_and_says_so():
    """A broken ML model must not silently produce a score."""
    import Agents.plugins.cardiovascular as cardio

    class BrokenModel:
        def predict_probability(self, data):
            raise RuntimeError("model file corrupt")

        def metadata(self):
            return {}

    original = cardio.CardioModel if hasattr(cardio, "CardioModel") else None
    import sys
    sys.modules["Agents.plugins.cardio_model"] = type(
        "M", (), {"CardioModel": BrokenModel}
    )
    try:
        risk = CardiovascularPlugin().predict_risk(AT_RISK)
    finally:
        del sys.modules["Agents.plugins.cardio_model"]
        if original is not None:
            cardio.CardioModel = original

    assert risk["prediction_method"] == "rule_based"
    assert risk["fallback_reason"]
    assert risk["risk_probability"] is not None


def test_flags_report_clinical_findings():
    flags = CardiovascularPlugin().flags(AT_RISK, {})
    assert "HYPERTENSION RANGE" in flags
    assert "HIGH LDL/CHOLESTEROL" in flags
    assert "HIGH LDL" in flags
    assert "SMOKING RISK FACTOR" in flags


def test_analysis_marks_abnormal_parameters():
    analysis = validate_parameters(AT_RISK, plugin=CardiovascularPlugin())
    assert analysis["abnormal_count"] >= 3
    assert "systolic_bp" in analysis["abnormal_parameters"]
    assert "total_cholesterol" in analysis["abnormal_parameters"]
    assert "HYPERTENSION RANGE" in analysis["flags"]


def test_analysis_reports_smoking_status():
    analysis = validate_parameters(AT_RISK, plugin=CardiovascularPlugin())
    assert analysis["parameters"]["smoker"]["value"] is True
    assert analysis["parameters"]["smoker"]["status"] == "Reported"


def test_healthy_analysis_has_no_abnormalities():
    analysis = validate_parameters(HEALTHY, plugin=CardiovascularPlugin())
    assert analysis["abnormal_count"] == 0


def test_red_flag_symptoms_are_flagged_as_emergency():
    result = CardiovascularPlugin().score_symptoms(["Crushing chest pain"])
    assert result["emergency"] is True
    assert result["severity_score"] == 1.0
    assert result["red_flags"] == ["Crushing chest pain"]


def test_cardiovascular_only_workflow_runs_end_to_end():
    from Agents.masterhealth import MasterHealthAgent

    agent = MasterHealthAgent()
    result = agent.process_health_data(AT_RISK, symptoms=["shortness of breath"])

    assert result["workflow_status"] == "completed"
    assert set(result["disease_results"]) == {"cardiovascular"}
    final = result["final_assessment"]
    assert final["disease"] == "cardiovascular"
    assert final["score"] is not None
    assert final["risk_level"] in ("Low", "Moderate", "High", "Critical")
    assert result["alert_result"]["alert"] is True
