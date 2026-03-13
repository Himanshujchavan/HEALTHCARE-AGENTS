from Agents.reportanalyzer import analyze_health_data
from Agents.masterhealth import _rule_based_risk, _to_probability


def test_risk_prediction_high():
    data = {
        "hba1c": 7.1,
        "glucose": 170,
        "bmi": 31,
        "age": 52,
    }

    analysis = analyze_health_data(data)
    risk = _rule_based_risk(analysis, data)

    probability = _to_probability(
        risk.get("risk_probability") or risk.get("risk_percentage")
    )

    assert probability > 0.5


def test_risk_prediction_low():
    data = {
        "hba1c": 5.0,
        "glucose": 90,
        "bmi": 22,
        "age": 25,
    }

    analysis = analyze_health_data(data)
    risk = _rule_based_risk(analysis, data)

    probability = _to_probability(
        risk.get("risk_probability") or risk.get("risk_percentage")
    )

    assert probability < 0.5
