"""Tests for the Explainability Agent."""

from Agents.explainer import Explainer


ASSESSMENT = {
    "disease": "diabetes",
    "risk_level": "High",
    "score": 0.71,
    "formula": "0.60*risk + 0.30*abnormal + 0.10*symptom",
    "abnormal_count": 2,
    "components": {
        "ml_component": 0.3,
        "abnormal_component": 0.3,
        "symptom_component": 0.05,
        "trend_component": 0.0,
    },
}


def test_explanation_reports_level_score_and_components():
    explanation = Explainer.explain(ASSESSMENT)
    assert "Diabetes" in explanation
    assert "High" in explanation
    assert "71.0%" in explanation
    assert "Model Risk: 0.3" in explanation
    assert "Abnormal Labs: 0.3" in explanation
    assert "Symptoms: 0.05" in explanation


def test_explanation_states_the_formula_used():
    explanation = Explainer.explain(ASSESSMENT)
    assert "0.60*risk + 0.30*abnormal + 0.10*symptom" in explanation


def test_explanation_mentions_a_completed_trend():
    explanation = Explainer.explain(
        ASSESSMENT,
        trend={"status": "completed", "direction": "rising", "visits_used": 3,
               "rising_but_normal": ["hba1c"]},
    )
    assert "rising" in explanation
    assert "hba1c" in explanation


def test_explanation_mentions_a_baseline_visit():
    explanation = Explainer.explain(
        ASSESSMENT, trend={"status": "insufficient_history", "visits_used": 1}
    )
    assert "Baseline visit" in explanation


def test_explanation_lists_alert_triggers():
    explanation = Explainer.explain(ASSESSMENT, triggers=["hba1c 7.20 >= 7.00"])
    assert "Alert Triggers" in explanation
    assert "hba1c 7.20 >= 7.00" in explanation


def test_explanation_handles_a_missing_assessment():
    assert Explainer.explain({}) == "No assessment data available to generate explanation."
