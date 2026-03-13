from Agents.reportanalyzer import analyze_health_data
from Agents.masterhealth import (
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
    # Any valid mapped risk level from the pipeline is acceptable
    assert alert["risk_level"] in ["Low", "Moderate", "High", "Critical"]
