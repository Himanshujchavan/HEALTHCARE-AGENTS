from Agents.reportanalyzer import analyze_health_data
from Agents.masterhealth import (
    MasterHealthAgent,
    _rule_based_risk,
    _to_probability,
    _derive_symptom_metrics,
    _score_to_risk_level,
)
from Agents.symptomchecker import analyze_symptoms
from Agents.alertsystem import AlertAgent


def test_full_agents_pipeline():

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

    # Report Analyzer
    analysis = analyze_health_data(health_data)
    abnormal_count = analysis["abnormal_count"]

    # Risk Predictor
    risk = _rule_based_risk(analysis, health_data)
    ml_probability = _to_probability(
        risk.get("risk_probability") or risk.get("risk_percentage")
    )

    # Symptom Checker
    symptom_result = analyze_symptoms(
        symptoms=symptoms,
        report_context=analysis,
        risk_context=risk,
        manual_text="",
        use_llm=False,
    )

    symptom_result.update(_derive_symptom_metrics(symptom_result))

    # Master fusion
    abnormal_weight = min(max(abnormal_count, 0) / 3.0, 1.0)
    symptom_score = symptom_result["severity_score"]

    final_score = (
        (ml_probability * 0.6)
        + (abnormal_weight * 0.3)
        + (symptom_score * 0.1)
    )

    risk_level = _score_to_risk_level(final_score)

    # Alert agent
    alert_agent = AlertAgent(use_llm=False)

    alert = alert_agent.process({
        "health_data": health_data,
        "analysis_result": analysis,
        "risk_result": risk,
        "diabetes_probability": final_score,
        "abnormal_parameters": abnormal_count,
        "symptom_score": symptom_score,
    })

    assert "report" in alert
    # The expected level for this case is fixed, not "any valid level".
    assert risk_level == "Critical"
    assert alert["risk_level"] == "Critical"


def test_orchestrated_pipeline_matches_the_manual_pipeline():
    """The LangGraph workflow reproduces the documented manual fusion."""
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

    analysis = analyze_health_data(health_data)
    risk = _rule_based_risk(analysis, health_data)
    ml_probability = _to_probability(
        risk.get("risk_probability") or risk.get("risk_percentage")
    )
    symptom_result = analyze_symptoms(
        symptoms=symptoms,
        report_context=analysis,
        risk_context=risk,
        manual_text="",
        use_llm=False,
    )
    symptom_result.update(_derive_symptom_metrics(symptom_result))

    expected_score = (
        (ml_probability * 0.6)
        + (min(analysis["abnormal_count"] / 3.0, 1.0) * 0.3)
        + (symptom_result["severity_score"] * 0.1)
    )

    agent = MasterHealthAgent()
    result = agent.process_health_data(health_data, symptoms=symptoms)
    final = result["final_assessment"]

    assert final["score"] == round(expected_score, 4)
    assert final["risk_level"] == _score_to_risk_level(expected_score)
    assert result["risk_level"] == final["risk_level"]
    assert result["alert_result"]["risk_level"] == final["risk_level"]
