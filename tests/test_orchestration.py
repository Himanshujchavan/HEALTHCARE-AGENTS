"""Tests for the multi-agent orchestration.

Covers the orchestration contract end to end: one shared typed state, one
input/output shape per agent, every applicable plugin invoked, a fusion that
does not lie about missing inputs, and explicit status for failed or skipped
agents.
"""

import pytest

from Agents import contracts
from Agents.contracts import (
    HealthWorkflowState,
    build_unified_response,
    fuse_results,
    normalize_risk_result,
    normalize_symptom_result,
    score_to_risk_level,
    to_probability,
)
from Agents.masterhealth import (
    MasterHealthAgent,
    _route_after_analysis,
    _route_after_differential,
    _route_after_merge,
    _route_after_symptoms,
    _route_after_trend,
    _route_entry,
    analyze_report_node,
    merge_results_node,
)
from Agents.plugins import registry


DIABETES_DATA = {"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45}
CARDIO_DATA = {
    "age": 55,
    "systolic_bp": 150,
    "total_cholesterol": 220,
    "smoker": True,
}
TWO_DISEASE_DATA = {**DIABETES_DATA, **CARDIO_DATA}


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED TYPED STATE
# ══════════════════════════════════════════════════════════════════════════════

def test_state_is_shared_between_contracts_and_orchestrator():
    """The graph state has one definition, imported by every participant."""
    from Agents import masterhealth

    assert masterhealth.HealthWorkflowState is HealthWorkflowState
    assert set(HealthWorkflowState.__annotations__) >= {
        "health_data",
        "disease_results",
        "agent_status",
        "history",
        "trend_result",
        "differential_result",
        "explanation",
    }


def test_state_carries_every_multi_disease_field():
    agent = MasterHealthAgent()
    result = agent.process_health_data(TWO_DISEASE_DATA)
    assert set(result["disease_results"]) == {"diabetes", "cardiovascular"}


# ══════════════════════════════════════════════════════════════════════════════
#  STANDARDIZED AGENT INPUTS AND OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def test_every_plugin_emits_the_same_risk_contract():
    required_keys = {
        "status",
        "disease",
        "risk_level",
        "risk_probability",
        "risk_score",
        "risk_percentage",
        "prediction_method",
        "risk_factors",
        "recommendations",
    }
    for name in registry.list_all():
        plugin = registry.get(name)
        data = DIABETES_DATA if name == "diabetes" else CARDIO_DATA
        for risk in (plugin.rule_risk(data), plugin.predict_risk(data)):
            normalized = normalize_risk_result(risk, disease=name)
            assert required_keys <= set(normalized)
            assert normalized["disease"] == name
            assert normalized["risk_level"] in ("Low", "Moderate", "High", "Critical")
            assert 0.0 <= normalized["risk_probability"] <= 1.0


def test_risk_contract_normalizes_foreign_payloads():
    """A payload from a different agent shape is coerced, not passed through."""
    normalized = normalize_risk_result(
        {"risk_level": "MEDIUM", "risk_probability": "72%"},
        disease="diabetes",
    )
    assert normalized["risk_level"] == "Moderate"
    assert normalized["risk_probability"] == 0.72
    assert normalized["risk_percentage"] == "72%"
    assert normalized["status"] == "completed"


def test_failed_risk_payload_is_reported_as_failed_not_scored():
    normalized = normalize_risk_result({"error": "model unavailable"}, disease="diabetes")
    assert normalized["status"] == "failed"
    assert normalized["risk_level"] == "Unavailable"
    assert normalized["risk_probability"] is None
    assert normalized["error"] == "model unavailable"


def test_symptom_contract_is_normalized_for_every_plugin():
    for name in registry.list_all():
        plugin = registry.get(name)
        normalized = normalize_symptom_result(plugin.score_symptoms(["chest pain"]))
        assert normalized["status"] == "completed"
        assert 0.0 <= normalized["severity_score"] <= 1.0
        assert "symptom_alignment" in normalized


def test_probability_and_level_helpers():
    assert to_probability("82%") == 0.82
    assert to_probability(0.4) == 0.4
    assert to_probability(82) == 0.82
    assert to_probability(None) is None
    assert score_to_risk_level(0.9) == "Critical"
    assert score_to_risk_level(0.7) == "High"
    assert score_to_risk_level(0.5) == "Moderate"
    assert score_to_risk_level(0.1) == "Low"


# ══════════════════════════════════════════════════════════════════════════════
#  THE MASTER AGENT INVOKES EVERY APPLICABLE PLUGIN
# ══════════════════════════════════════════════════════════════════════════════

def test_every_applicable_plugin_is_invoked():
    agent = MasterHealthAgent()
    result = agent.process_health_data(TWO_DISEASE_DATA)

    expected = {p.name for p in registry.applicable(TWO_DISEASE_DATA)}
    assert expected == {"diabetes", "cardiovascular"}
    assert set(result["disease_results"]) == expected

    for disease in expected:
        entry = result["disease_results"][disease]
        for stage in ("analysis", "risk", "symptom", "trend", "final_assessment"):
            assert stage in entry, f"{disease} is missing {stage}"
        assert entry["final_assessment"]["score"] is not None
        assert entry["agent_status"]["analysis"]["status"] == "completed"
        assert entry["agent_status"]["risk"]["status"] == "completed"


def test_only_applicable_plugins_run():
    agent = MasterHealthAgent()
    result = agent.process_health_data(DIABETES_DATA)
    assert set(result["disease_results"]) == {"diabetes"}


def test_agent_status_is_preserved_across_nodes():
    """Every agent's status survives to the end of the run."""
    agent = MasterHealthAgent()
    result = agent.process_health_data(TWO_DISEASE_DATA)

    for stage in ("analysis", "risk", "symptom", "trend", "differential", "fusion"):
        assert stage in result["agent_status"], f"missing status for {stage}"
        assert result["agent_status"][stage]["status"]


# ══════════════════════════════════════════════════════════════════════════════
#  RELIABLE RESULT FUSION
# ══════════════════════════════════════════════════════════════════════════════

def test_fusion_requires_analysis_and_risk():
    """Without the mandatory components the disease is explicitly unavailable."""
    analysis = {"abnormal_count": 1}
    risk = {"risk_probability": 0.5}

    no_analysis = fuse_results(None, risk, {"severity_score": 0.2})
    no_risk = fuse_results(analysis, None, {"severity_score": 0.2})

    for fusion in (no_analysis, no_risk):
        assert fusion["status"] == "unavailable"
        assert fusion["score"] is None
        assert fusion["risk_level"] == "Unavailable"


def test_fusion_uses_the_base_formula_without_history():
    fusion = fuse_results(
        analysis={"abnormal_count": 3},
        risk={"risk_probability": 0.82},
        symptom={"severity_score": 0.0, "status": "skipped"},
        trend={"status": "insufficient_history", "trend_score": 0.0},
    )
    assert fusion["formula"] == "0.60*risk + 0.30*abnormal + 0.10*symptom"
    assert fusion["score"] == pytest.approx(0.6 * 0.82 + 0.3 * 1.0)
    assert "trend" not in fusion["effective_weights"]


def test_fusion_uses_the_extended_formula_with_a_computed_trend():
    fusion = fuse_results(
        analysis={"abnormal_count": 3},
        risk={"risk_probability": 0.82},
        symptom={"severity_score": 0.0, "status": "skipped"},
        trend={"status": "completed", "trend_score": 0.5},
    )
    assert fusion["formula"] == "0.50*risk + 0.25*abnormal + 0.10*symptom + 0.15*trend"
    assert fusion["score"] == pytest.approx(
        0.5 * 0.82 + 0.25 * 1.0 + 0.1 * 0.0 + 0.15 * 0.5
    )


def test_fusion_does_not_inflate_the_score_when_an_agent_fails():
    """A failed component is dropped, not silently redistributed."""
    complete = fuse_results(
        analysis={"abnormal_count": 3},
        risk={"risk_probability": 0.82},
        symptom={"severity_score": 0.4, "status": "completed"},
    )
    degraded = fuse_results(
        analysis={"abnormal_count": 3},
        risk={"risk_probability": 0.82},
        symptom={"error": "symptom checker crashed", "status": "failed"},
    )

    assert degraded["status"] == "completed_with_errors"
    assert degraded["missing_components"] == ["symptom"]
    assert degraded["coverage"] < 1.0
    assert degraded["score"] < complete["score"]


def test_fusion_reports_skipped_components():
    fusion = fuse_results(
        analysis={"abnormal_count": 0},
        risk={"risk_probability": 0.1},
        symptom={"severity_score": 0.0, "status": "skipped"},
    )
    assert fusion["skipped_components"] == ["symptom"]
    assert fusion["missing_components"] == []
    assert fusion["status"] == "completed"


def test_merge_node_reports_unavailable_diseases():
    """A disease whose risk failed is listed, not silently dropped."""
    state = {
        "health_data": TWO_DISEASE_DATA,
        "steps": {},
        "disease_results": {
            "diabetes": {
                "analysis": {"abnormal_count": 2},
                "risk": {"risk_probability": 0.4},
                "symptom": {"severity_score": 0.1, "status": "completed"},
            },
            "cardiovascular": {
                "analysis": {"abnormal_count": 2},
                "risk": {"error": "risk engine down", "status": "failed"},
                "symptom": {"severity_score": 0.1, "status": "completed"},
            },
        },
    }
    result = merge_results_node(state)

    assert result["final_assessment"]["disease"] == "diabetes"
    assert result["final_assessment"]["unavailable_diseases"] == ["cardiovascular"]
    assert "cardiovascular" in result["final_assessment"]["all_diseases"]
    assert result["steps"]["6_result_fusion"]["status"] == "completed_with_errors"


def test_merge_node_does_not_leak_one_disease_trend_into_another():
    """Each disease is fused with its own trend."""
    state = {
        "health_data": TWO_DISEASE_DATA,
        "steps": {},
        "trend_result": {"status": "completed", "trend_score": 1.0},
        "disease_results": {
            "diabetes": {
                "analysis": {"abnormal_count": 0},
                "risk": {"risk_probability": 0.2},
                "symptom": {"severity_score": 0.0, "status": "skipped"},
                "trend": {"status": "insufficient_history", "trend_score": 0.0},
            },
            "cardiovascular": {
                "analysis": {"abnormal_count": 0},
                "risk": {"risk_probability": 0.2},
                "symptom": {"severity_score": 0.0, "status": "skipped"},
                "trend": {"status": "completed", "trend_score": 0.8},
            },
        },
    }
    result = merge_results_node(state)
    fusions = result["final_assessment"]["all_diseases"]

    assert "trend" not in fusions["diabetes"]["effective_weights"]
    assert fusions["diabetes"]["score"] == pytest.approx(0.6 * 0.2)
    assert fusions["cardiovascular"]["score"] == pytest.approx(
        0.5 * 0.2 + 0.25 * 0.0 + 0.1 * 0.0 + 0.15 * 0.8
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CONDITIONAL ROUTING
# ══════════════════════════════════════════════════════════════════════════════

def test_routing_for_missing_inputs_and_failures():
    ok = {"workflow_status": "in_progress"}
    failed = {"workflow_status": "failed"}

    assert _route_entry({"input_type": "data"}) == "analyze"
    assert _route_entry({"input_type": "pdf"}) == "extract_pdf"

    assert _route_after_analysis(failed) == "finalize"
    assert _route_after_analysis(ok) == "predict_risk"
    assert _route_after_analysis({**ok, "include_risk": False}) == "check_symptoms"

    assert _route_after_symptoms(failed) == "finalize"
    assert _route_after_symptoms(ok) == "trend"
    assert _route_after_trend(failed) == "finalize"
    assert _route_after_trend(ok) == "differential"
    assert _route_after_differential(failed) == "finalize"
    assert _route_after_differential(ok) == "merge_results"
    assert _route_after_merge(failed) == "finalize"
    assert _route_after_merge(ok) == "explain"


def test_analysis_fails_when_no_plugin_is_applicable():
    result = analyze_report_node(
        {"health_data": {"age": 30}, "steps": {}, "disease_results": {}}
    )
    assert result["workflow_status"] == "failed"
    assert "No applicable disease plugins" in result["error"]


def test_workflow_reports_failure_when_nothing_is_applicable():
    agent = MasterHealthAgent()
    result = agent.process_health_data({"age": 30})
    assert result["workflow_status"] == "failed"
    assert result["error"]


# ══════════════════════════════════════════════════════════════════════════════
#  NO MISLEADING ASSESSMENTS
# ══════════════════════════════════════════════════════════════════════════════

def test_absent_symptoms_are_reported_as_skipped_not_completed():
    agent = MasterHealthAgent()
    result = agent.process_health_data(DIABETES_DATA)

    symptom_status = result["disease_results"]["diabetes"]["agent_status"]["symptom"]
    assert symptom_status["status"] == "skipped"
    assert "No symptoms" in symptom_status["error"]

    fusion = result["final_assessment"]
    assert fusion["symptom_score"] == 0.0
    assert "symptom" in fusion["skipped_components"]


def test_symptom_agent_is_completed_when_symptoms_are_supplied():
    agent = MasterHealthAgent()
    result = agent.process_health_data(
        DIABETES_DATA, symptoms=["Fatigue / Low energy", "Polyuria (frequent urination)"]
    )
    symptom_status = result["disease_results"]["diabetes"]["agent_status"]["symptom"]
    assert symptom_status["status"] == "completed"


def test_explanation_is_stored_on_the_assessment():
    agent = MasterHealthAgent()
    result = agent.process_health_data(DIABETES_DATA)
    assert result["final_assessment"]["explanation"]
    assert result["explanation"] == result["final_assessment"]["explanation"]


def test_unified_response_marks_non_applicable_diseases():
    agent = MasterHealthAgent()
    result = agent.process_health_data(DIABETES_DATA)
    unified = build_unified_response(result, assessment_id=1)

    assert unified["assessments"]["diabetes"]["status"] != "not_applicable"
    assert unified["assessments"]["cardiovascular"]["status"] == "not_applicable"
    assert unified["assessments"]["cardiovascular"]["reason"]
    assert unified["model_disclaimer"]


def test_unified_response_exposes_agent_status():
    agent = MasterHealthAgent()
    result = agent.process_health_data(TWO_DISEASE_DATA)
    unified = build_unified_response(result, assessment_id=1)

    for disease in ("diabetes", "cardiovascular"):
        statuses = unified["assessments"][disease]["agent_status"]
        assert statuses["analysis"]["status"] == "completed"
        assert statuses["risk"]["status"] == "completed"
    assert unified["agent_status"]
