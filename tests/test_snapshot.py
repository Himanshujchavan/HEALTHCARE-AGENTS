import pytest
from Agents.reportanalyzer import analyze_health_data
from Agents.masterhealth import (
    _rule_based_risk,
    _to_probability,
    _derive_symptom_metrics,
)
from Agents.symptomchecker import analyze_symptoms
from Agents.alertsystem import AlertAgent

def test_snapshot_regression():
    """
    Snapshot regression test for the target case:
    hba1c=6.8, glucose=148, bmi=29, age=45
    """
    health_data = {
        "hba1c": 6.8,
        "glucose": 148,
        "bmi": 29,
        "age": 45,
    }
    symptoms = [
        "Fatigue / Low energy",
        "Polyuria (frequent urination)",
    ]

    # 1. Report Analyzer
    analysis = analyze_health_data(health_data)
    abnormal_count = analysis["abnormal_count"]

    # 2. Risk Predictor
    risk = _rule_based_risk(analysis, health_data)
    ml_probability = _to_probability(
        risk.get("risk_probability") or risk.get("risk_percentage")
    )

    # 3. Symptom Checker
    symptom_result = analyze_symptoms(
        symptoms=symptoms,
        report_context=analysis,
        risk_context=risk,
        manual_text="",
        use_llm=False,
    )
    symptom_result.update(_derive_symptom_metrics(symptom_result))

    # 4. Master fusion (Default: 0.6 / 0.3 / 0.1)
    abnormal_weight = min(max(abnormal_count, 0) / 3.0, 1.0)
    symptom_score = symptom_result["severity_score"]

    final_score = (
        (ml_probability * 0.6)
        + (abnormal_weight * 0.3)
        + (symptom_score * 0.1)
    )

    # Expected values based on current logic:
    # - hba1c 6.8 -> high
    # - glucose 148 -> high
    # - bmi 29 -> high
    # - age 45 -> medium
    # Abnormal count = 3 -> abnormal_weight = 1.0
    # ML Rule risk for these values typically lands in "Moderate" or "High"
    # Symptom score for fatigue/polyuria typically ~0.5-0.7

    # We assert against the current actual pipeline results to lock the baseline.
    # These values are derived from running the current pipeline.
    expected_score = 0.892  # Actual current output: 0.8919999999999999

    assert final_score == pytest.approx(expected_score, abs=0.01)
