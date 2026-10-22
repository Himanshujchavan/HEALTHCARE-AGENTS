"""Tests for the Differential Agent."""

from Agents.differentialagent import DifferentialAgent


def _results(diabetes_score, cardio_score, diabetes_symptom=0.0, cardio_symptom=0.0):
    return {
        "diabetes": {
            "risk": {"risk_probability": diabetes_score},
            "symptom": {"severity_score": diabetes_symptom},
        },
        "cardiovascular": {
            "risk": {"risk_probability": cardio_score},
            "symptom": {"severity_score": cardio_symptom},
        },
    }


def test_single_disease_is_skipped():
    result = DifferentialAgent.run_differential(
        {"diabetes": {"risk": {"risk_probability": 0.5}}}, ["fatigue"]
    )
    assert result is None


def test_no_diseases_is_skipped():
    assert DifferentialAgent.run_differential({}, []) is None


def test_ranking_puts_the_higher_scoring_disease_first():
    result = DifferentialAgent.run_differential(
        _results(diabetes_score=0.9, cardio_score=0.2), ["fatigue"]
    )
    assert result["primary"] == "diabetes"
    assert [alt["disease"] for alt in result["alternatives"]] == ["cardiovascular"]
    assert result["alternatives"][0]["score"] < result["alternatives"][0]["risk_probability"] + 0.4


def test_symptom_overlap_breaks_a_tie():
    """With equal risk, the disease matching more symptoms ranks higher."""
    result = DifferentialAgent.run_differential(
        _results(0.5, 0.5, diabetes_symptom=0.0, cardio_symptom=1.0),
        ["chest pain"],
    )
    assert result["primary"] == "cardiovascular"


def test_shared_symptoms_are_intersected():
    result = DifferentialAgent.run_differential(
        _results(0.5, 0.5),
        ["fatigue", "chest pain", "dizziness"],
    )
    assert "fatigue" in result["shared_symptoms"]
    assert "chest pain" not in result["shared_symptoms"]


def test_workflow_produces_a_differential_for_two_diseases():
    from Agents.masterhealth import MasterHealthAgent

    agent = MasterHealthAgent()
    result = agent.process_health_data(
        {
            "hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 55,
            "systolic_bp": 165, "total_cholesterol": 250, "smoker": True,
        },
        symptoms=["fatigue", "chest pain"],
    )
    differential = result["differential_result"]
    assert differential is not None
    assert differential["primary"] in ("diabetes", "cardiovascular")
    assert len(differential["alternatives"]) == 1


def test_workflow_skips_the_differential_for_one_disease():
    from Agents.masterhealth import MasterHealthAgent

    agent = MasterHealthAgent()
    result = agent.process_health_data(
        {"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45},
        symptoms=["fatigue"],
    )
    assert result["differential_result"] is None
    assert result["agent_status"]["differential"]["status"] == "skipped"
