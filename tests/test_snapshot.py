import pytest
from Agents.masterhealth import MasterHealthAgent

def test_diabetes_snapshot_with_symptoms():
    test_data = {"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45}
    symptoms = [
        "Fatigue / Low energy",
        "Polyuria (frequent urination)",
    ]

    agent = MasterHealthAgent()
    result = agent.process_health_data(test_data, symptoms=symptoms)
    final = result.get("final_assessment", {})

    assert final.get("ml_probability") == 0.82
    assert final.get("abnormal_count") == 3
    assert pytest.approx(final.get("score"), abs=0.005) == 0.892
    assert final.get("risk_level") == "Critical"


def test_diabetes_snapshot_without_symptoms():
    test_data = {"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45}

    agent = MasterHealthAgent()
    result = agent.process_health_data(test_data)
    final = result.get("final_assessment", {})

    assert final.get("ml_probability") == 0.82
    assert final.get("abnormal_count") == 3
    assert pytest.approx(final.get("score"), abs=0.005) == 0.792
    assert final.get("risk_level") == "High"
