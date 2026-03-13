from Agents.alertsystem import AlertAgent


def test_alert_trigger_high_risk():

    alert_agent = AlertAgent(use_llm=False)

    data = {
        "health_data": {
            "hba1c": 7.0,
            "glucose": 165,
            "bmi": 30,
            "age": 50,
        },
        "analysis_result": {"abnormal_count": 3},
        "risk_result": {"risk_level": "High"},
        "diabetes_probability": 0.85,
        "abnormal_parameters": 3,
        "symptom_score": 0.2,
    }

    result = alert_agent.process(data)

    assert result["alert"] is True
    assert "report" in result


def test_alert_low_risk():

    alert_agent = AlertAgent(use_llm=False)

    data = {
        "health_data": {
            "hba1c": 5.1,
            "glucose": 90,
            "bmi": 22,
            "age": 25,
        },
        "analysis_result": {"abnormal_count": 0},
        "risk_result": {"risk_level": "Low"},
        "diabetes_probability": 0.1,
        "abnormal_parameters": 0,
        "symptom_score": 0.0,
    }

    result = alert_agent.process(data)

    assert result["alert"] is False
