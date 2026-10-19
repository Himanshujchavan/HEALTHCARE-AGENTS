"""Data contracts and functional entry points for sub-agents.

History Record Schema:
    {
        "created_at": str (ISO 8601),
        "params": Dict[str, float] (e.g. {"hba1c": 6.8, "glucose": 148, ...}),
        "risk_score": float (final assessment score 0-1 or 0-100)
    }

Disease Results Schema:
    {
        "<disease_name>": {
            "analysis": Dict[str, Any],
            "risk": Dict[str, Any],
            "symptom": Dict[str, Any],
            "final_assessment": Dict[str, Any]
        }
    }
"""

from typing import List, Dict, Any, Optional
from Agents.trendagent import TrendAgent
from Agents.differentialagent import DifferentialAgent
from Agents.explainer import Explainer

def run_trend(history: List[Dict], plugin: Any, current: Dict) -> Dict[str, Any]:
    """Calculate trend score based on historical records.
    Returns: {trend_score, direction, rising_but_normal, visits_used, status}
    """
    return TrendAgent.run_trend(history, plugin, current)

def run_differential(disease_results: Dict[str, Any], symptoms: List[str]) -> Optional[Dict[str, Any]]:
    """Analyze symptoms across applicable diseases to find the primary match.
    Returns: {primary, alternatives: [{disease, score}], shared_symptoms}
    """
    return DifferentialAgent.run_differential(disease_results, symptoms)

def explain(final_assessment: Dict[str, Any], trend: Optional[Dict] = None, triggers: Optional[List[str]] = None) -> str:
    """Generate a human-readable explanation of the risk score."""
    return Explainer.explain(final_assessment, trend, triggers)


def build_unified_response(
    workflow: Dict[str, Any],
    assessment_id: Optional[int] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Expose one stable, frontend-facing shape for every workflow execution."""
    disease_results = workflow.get("disease_results") or {}
    assessments: Dict[str, Any] = {}

    for disease in ("diabetes", "cardiovascular"):
        entry = disease_results.get(disease)
        if not isinstance(entry, dict):
            assessments[disease] = {
                "status": "unavailable",
                "risk_level": "Unavailable",
                "score": None,
                "factors": [],
            }
            continue

        analysis = entry.get("analysis") or {}
        risk = entry.get("risk") or {}
        final = entry.get("final_assessment") or {}
        factors = list(analysis.get("abnormal_parameters") or [])
        factors.extend(risk.get("triggers") or [])
        assessments[disease] = {
            "status": "completed" if final or risk else "partial",
            "risk_level": final.get("risk_level") or risk.get("risk_level") or "Unavailable",
            "score": final.get("score", risk.get("score")),
            "factors": list(dict.fromkeys(str(item) for item in factors)),
            "analysis": analysis,
            "risk": risk,
            "symptoms": entry.get("symptom"),
            "trend": entry.get("trend"),
            "final_assessment": final,
        }

    alert_result = workflow.get("alert_result") or {}
    alerts = []
    if isinstance(alert_result, dict) and alert_result.get("alert") is not None:
        alerts.append({
            "severity": alert_result.get("risk_level", "Unknown"),
            "alert": bool(alert_result.get("alert")),
            "explanation": alert_result.get("report") or alert_result.get("notification"),
            "recommended_next_steps": alert_result.get("notification"),
            "triggers": alert_result.get("triggers") or [],
        })

    differential = workflow.get("differential_result")
    if not isinstance(differential, dict) or not differential:
        differential = {"status": "unavailable", "results": []}
    else:
        differential = {"status": "completed", "results": differential}

    trend = workflow.get("trend_result")
    if not isinstance(trend, dict):
        trend = {"status": "unavailable", "results": []}
    elif trend.get("status") not in {"completed", "insufficient_history"}:
        trend = {"status": "failed", "results": []}

    response = {
        "assessment_id": assessment_id,
        "status": workflow.get("workflow_status", "unknown"),
        "assessments": assessments,
        "differential": differential,
        "trend": trend,
        "alerts": alerts,
        "explanation": workflow.get("explanation") or {},
        "created_at": created_at or workflow.get("timestamp"),
        "model_disclaimer": "These are model predictions and risk signals, not confirmed medical diagnoses.",
    }
    return response
