"""Shared data contracts for the multi-agent health workflow.

This module is the single source of truth for:

  * the typed LangGraph state (``HealthWorkflowState``)
  * the standardized agent status vocabulary
  * the standardized risk / symptom / analysis envelopes
  * the result-fusion function
  * the frontend-facing unified response builder

Every agent that crosses a node boundary speaks these contracts, so the
orchestrator never has to guess which shape a sub-agent returned.

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
            "trend": Dict[str, Any],
            "final_assessment": Dict[str, Any],
            "agent_status": Dict[str, Dict[str, Any]]
        }
    }
"""

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict

from Agents.trendagent import TrendAgent
from Agents.differentialagent import DifferentialAgent
from Agents.explainer import Explainer
from utils.helpers import normalize_level


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT STATUS VOCABULARY
#
#  Every agent outcome is reported with one of these statuses. A step that
#  could not run is never reported as "completed" — that is what keeps an
#  unavailable agent from silently producing a misleading assessment.
# ══════════════════════════════════════════════════════════════════════════════

STATUS_COMPLETED = "completed"
STATUS_COMPLETED_WITH_ERRORS = "completed_with_errors"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_UNAVAILABLE = "unavailable"
STATUS_NOT_APPLICABLE = "not_applicable"

#: Statuses that mean "the agent produced a usable result".
USABLE_STATUSES = frozenset({STATUS_COMPLETED, STATUS_COMPLETED_WITH_ERRORS})

#: Statuses that mean "the agent did not produce a result".
MISSING_STATUSES = frozenset(
    {STATUS_FAILED, STATUS_UNAVAILABLE, STATUS_NOT_APPLICABLE}
)


def agent_status(status: str, error: Optional[str] = None) -> Dict[str, Any]:
    """Build the common status shape used by graph steps and disease agents."""
    result: Dict[str, Any] = {"status": status}
    if error:
        result["error"] = error
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  TYPED WORKFLOW STATE
# ══════════════════════════════════════════════════════════════════════════════

class HealthWorkflowState(TypedDict):
    """Typed state that flows through the LangGraph workflow.

    ``disease_results`` holds the per-disease outputs keyed by plugin name
    (e.g. ``"diabetes"``, ``"cardiovascular"``), while ``analysis_result`` /
    ``risk_result`` / ``symptom_result`` keep the primary (highest scoring)
    disease so existing routes, tests and the frontend keep working.
    """
    health_data: Dict[str, Any]
    symptoms: List[str]
    manual_text: Optional[str]
    include_risk: bool
    include_symptom: bool
    include_alert: bool
    input_type: str                 # "data" | "pdf"
    pdf_path: Optional[str]
    use_llm: bool
    analysis_result: Optional[Dict]
    risk_result: Optional[Dict]
    symptom_result: Optional[Dict]
    final_assessment: Optional[Dict]
    alert_result: Optional[Dict]
    workflow_status: str
    error: Optional[str]
    steps: Dict[str, Any]
    timestamp: str
    input_file: Optional[str]
    # Multi-disease support
    disease_results: Dict[str, Dict[str, Any]]
    agent_status: Dict[str, Dict[str, Any]]
    history: Optional[List[Dict]]
    trend_result: Optional[Dict]
    differential_result: Optional[Dict]
    explanation: Optional[str]


# ══════════════════════════════════════════════════════════════════════════════
#  VALUE NORMALIZATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def to_probability(value: Any) -> Optional[float]:
    """Parse probability values from 0.82 / "82%" / 82 formats into [0,1]."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            try:
                return float(text[:-1].strip()) / 100.0
            except ValueError:
                return None
        try:
            num = float(text)
        except ValueError:
            return None
    else:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None

    if num > 1:
        return max(0.0, min(1.0, num / 100.0))
    return max(0.0, min(1.0, num))


def score_to_risk_level(score: float) -> str:
    """Map a numeric fusion score to the standardized risk level set."""
    if score >= 0.80:
        return "Critical"
    elif score >= 0.65:
        return "High"
    elif score >= 0.40:
        return "Moderate"
    return "Low"


def _component_status(result: Optional[Dict[str, Any]]) -> str:
    """Derive the status of one fusion component from its raw payload."""
    if not result:
        return STATUS_UNAVAILABLE
    if result.get("error"):
        return STATUS_FAILED
    status = result.get("status")
    if status in (
        STATUS_COMPLETED,
        STATUS_COMPLETED_WITH_ERRORS,
        STATUS_FAILED,
        STATUS_SKIPPED,
        STATUS_UNAVAILABLE,
        STATUS_NOT_APPLICABLE,
    ):
        return status
    # Trend-agent specific statuses mapped onto the shared vocabulary.
    if status == "insufficient_history":
        return STATUS_SKIPPED
    if status == "no_plugins":
        return STATUS_NOT_APPLICABLE
    return STATUS_COMPLETED


# ══════════════════════════════════════════════════════════════════════════════
#  STANDARDIZED AGENT ENVELOPES
# ══════════════════════════════════════════════════════════════════════════════

def normalize_risk_result(
    raw: Optional[Dict[str, Any]],
    *,
    disease: Optional[str] = None,
    method: Optional[str] = None,
) -> Dict[str, Any]:
    """Coerce any agent risk payload into the canonical risk contract.

    The canonical risk contract is::

        {
            "status": "completed",
            "disease": "diabetes",
            "risk_level": "Low|Moderate|High|Critical",
            "risk_probability": float | None,   # 0-1
            "risk_score": float | None,          # 0-100
            "risk_percentage": str | None,
            "prediction_method": str,
            "risk_factors": [...],
            "recommendations": [...],
            "summary": str,
        }
    """
    if not isinstance(raw, dict) or raw.get("error"):
        error = (
            raw.get("error")
            if isinstance(raw, dict)
            else "Risk predictor returned an invalid result"
        )
        return {
            "status": STATUS_FAILED,
            "disease": disease,
            "risk_level": "Unavailable",
            "risk_probability": None,
            "risk_score": None,
            "risk_percentage": None,
            "prediction_method": method or "unknown",
            "risk_factors": [],
            "recommendations": [],
            "summary": "",
            "error": error,
        }

    probability = to_probability(
        raw.get("risk_probability")
        or raw.get("risk_percentage")
        or raw.get("score")
    )
    risk_score = raw.get("risk_score")
    if risk_score is None and probability is not None:
        risk_score = round(probability * 100, 2)

    percentage = raw.get("risk_percentage")
    if not percentage and probability is not None:
        percentage = f"{int(round(probability * 100))}%"

    extra_keys = (
        "model_metadata",
        "model_confidence",
        "model_variant",
        "model_features",
        "model_missing_optional_features",
        "model_metrics",
        "model_target",
        "model_limitations",
        "disclaimer",
        "response",
        "success",
        "triggers",
    )

    return {
        "status": raw.get("status", STATUS_COMPLETED),
        "disease": disease,
        "risk_level": normalize_level(raw.get("risk_level") or raw.get("level") or "Low"),
        "risk_probability": probability,
        "risk_score": risk_score,
        "risk_percentage": percentage,
        "prediction_method": (
            raw.get("prediction_method") or raw.get("method") or method or "rule_based"
        ),
        "risk_factors": list(raw.get("risk_factors") or []),
        "recommendations": list(raw.get("recommendations") or []),
        "summary": raw.get("summary") or raw.get("response") or "",
        **({key: raw[key] for key in extra_keys if key in raw}),
        **({"fallback_reason": raw["fallback_reason"]} if raw.get("fallback_reason") else {}),
    }


def normalize_symptom_result(
    raw: Optional[Dict[str, Any]],
    *,
    status: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Coerce any agent symptom payload into the canonical symptom contract.

    The raw payload is preserved (the alert agent reads ``symptom_mapping``)
    and the canonical fields are added on top.
    """
    if not isinstance(raw, dict):
        raw = {}

    severity = raw.get("severity_score")
    if severity is None:
        severity = raw.get("severity", 0.0)
    try:
        severity = max(0.0, min(1.0, float(severity)))
    except (TypeError, ValueError):
        severity = 0.0

    resolved_status = status or raw.get("status") or STATUS_COMPLETED
    if raw.get("error"):
        resolved_status = STATUS_FAILED

    normalized = dict(raw)
    normalized["severity_score"] = round(severity, 4)
    normalized["severity"] = round(severity, 4)
    normalized.setdefault("symptom_alignment", _alignment_for_severity(severity))
    normalized.setdefault("top_hypothesis", raw.get("input_summary", {}).get("top_hypothesis")
                          if isinstance(raw.get("input_summary"), dict) else None)
    normalized["status"] = resolved_status
    if reason:
        normalized["reason"] = reason
    return normalized


def _alignment_for_severity(severity: float) -> str:
    if severity >= 0.7:
        return "high"
    if severity >= 0.35:
        return "moderate"
    return "low"


def normalize_analysis_result(
    raw: Optional[Dict[str, Any]],
    *,
    disease: Optional[str] = None,
) -> Dict[str, Any]:
    """Coerce an analyzer payload into the canonical analysis contract.

    ``flags`` is the disease-neutral key; ``endocrine_flags`` is kept as an
    alias so existing diabetes consumers keep working.
    """
    if not isinstance(raw, dict):
        raw = {}

    flags = list(raw.get("flags") or raw.get("endocrine_flags") or [])
    normalized = dict(raw)
    normalized["flags"] = flags
    normalized["endocrine_flags"] = flags
    normalized["abnormal_count"] = int(raw.get("abnormal_count", 0) or 0)
    normalized["abnormal_parameters"] = list(raw.get("abnormal_parameters") or [])
    normalized["parameters"] = raw.get("parameters") or {}
    normalized["status"] = raw.get("status", STATUS_COMPLETED)
    if disease:
        normalized["disease"] = disease
    return normalized


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT FUSION
# ══════════════════════════════════════════════════════════════════════════════

#: Weights used when no longitudinal trend could be computed.
FUSION_WEIGHTS_BASE = {"ml": 0.60, "abnormal": 0.30, "symptom": 0.10}
#: Weights used once a trend over two or more prior visits is available.
FUSION_WEIGHTS_WITH_TREND = {"ml": 0.50, "abnormal": 0.25, "symptom": 0.10, "trend": 0.15}

FORMULA_BASE = "0.60*risk + 0.30*abnormal + 0.10*symptom"
FORMULA_WITH_TREND = "0.50*risk + 0.25*abnormal + 0.10*symptom + 0.15*trend"


def fuse_results(
    analysis: Optional[Dict[str, Any]],
    risk: Optional[Dict[str, Any]],
    symptom: Optional[Dict[str, Any]],
    trend: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fuse lab, model-risk, symptom and trend signals into one assessment.

    Rules that make the result trustworthy:

    * ``analysis`` and ``risk`` are mandatory. Without them the disease gets
      an explicit ``unavailable`` assessment instead of a guessed score.
    * A component that could not run is never silently renormalized away.
      Its nominal weight is kept, the component contributes zero, and the
      result is reported as ``partial`` with the missing component named.
    * The extended formula is used only when a trend was actually computed.
    """
    component_status = {
        "analysis": _component_status(analysis),
        "risk": _component_status(risk),
        "symptom": _component_status(symptom),
        "trend": _component_status(trend),
    }

    if component_status["analysis"] in MISSING_STATUSES or \
            component_status["risk"] in MISSING_STATUSES:
        return {
            "status": STATUS_UNAVAILABLE,
            "risk_level": "Unavailable",
            "score": None,
            "component_status": component_status,
            "missing_components": [
                name for name, status in component_status.items()
                if status in MISSING_STATUSES
            ],
            "reason": "Required analysis or risk result is unavailable.",
        }

    abnormal_count = int((analysis or {}).get("abnormal_count", 0) or 0)
    abnormal_weight = min(max(abnormal_count, 0) / 3.0, 1.0)

    ml_probability = to_probability(
        (risk or {}).get("risk_probability")
        or (risk or {}).get("risk_percentage")
        or ((risk or {}).get("risk_score") / 100.0 if (risk or {}).get("risk_score") is not None else None)
    )
    if ml_probability is None:
        ml_probability = 0.0

    symptom_score = (symptom or {}).get("severity_score")
    if symptom_score is None:
        symptom_score = (symptom or {}).get("severity", 0.0)
    try:
        symptom_score = float(symptom_score)
    except (TypeError, ValueError):
        symptom_score = 0.0
    symptom_score = max(0.0, min(1.0, symptom_score))

    trend_score = 0.0
    if component_status["trend"] in USABLE_STATUSES:
        try:
            trend_score = float((trend or {}).get("trend_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            trend_score = 0.0

    # The extended formula applies only when a trend was actually computed.
    use_extended = component_status["trend"] == STATUS_COMPLETED
    if use_extended:
        weights = dict(FUSION_WEIGHTS_WITH_TREND)
        formula = FORMULA_WITH_TREND
    else:
        weights = dict(FUSION_WEIGHTS_BASE)
        formula = FORMULA_BASE

    components = {
        "ml": ml_probability,
        "abnormal": abnormal_weight,
        "symptom": symptom_score,
        "trend": trend_score,
    }

    # A component contributes its weight when it ran or was legitimately
    # skipped (e.g. no symptoms reported -> zero contribution). A component
    # that failed or was unavailable is excluded from the score entirely.
    applied = {
        name: weight
        for name, weight in weights.items()
        if component_status.get(name) not in MISSING_STATUSES
    }
    missing_components = [
        name for name in weights if name not in applied
    ]
    skipped_components = [
        name for name in weights
        if component_status.get(name) == STATUS_SKIPPED
    ]

    final_score = round(
        sum(components[name] * weight for name, weight in applied.items()), 4
    )
    coverage = round(sum(applied.values()), 4)

    # A component that failed leaves the score resting on less than the
    # full formula, so the assessment is only partial. A component that was
    # legitimately skipped (no symptoms supplied, no history) still means the
    # documented formula was applied in full.
    status = STATUS_COMPLETED_WITH_ERRORS if missing_components else STATUS_COMPLETED

    return {
        "risk_level": normalize_level(score_to_risk_level(final_score)),
        "status": status,
        "score": final_score,
        "ml_probability": ml_probability,
        "abnormal_count": abnormal_count,
        "abnormal_count_weight": round(abnormal_weight, 4),
        "symptom_score": round(symptom_score, 4),
        "trend_score": round(trend_score, 4),
        "formula": formula,
        "effective_weights": applied,
        "coverage": coverage,
        "missing_components": missing_components,
        "skipped_components": skipped_components,
        "component_status": component_status,
        "components": {
            "ml_component": round(ml_probability * weights["ml"], 4),
            "abnormal_component": round(abnormal_weight * weights["abnormal"], 4),
            "symptom_component": round(symptom_score * weights["symptom"], 4),
            "trend_component": round(trend_score * weights.get("trend", 0.0), 4),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════════════

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


def explain(final_assessment: Dict[str, Any], trend: Optional[Dict[str, Any]] = None, triggers: Optional[List[str]] = None) -> str:
    """Generate a human-readable explanation of the risk score."""
    return Explainer.explain(final_assessment, trend, triggers)


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED RESPONSE
# ══════════════════════════════════════════════════════════════════════════════

def build_unified_response(
    workflow: Dict[str, Any],
    assessment_id: Optional[int] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Expose one stable, frontend-facing shape for every workflow execution.

    Every registered disease gets an entry. Diseases that were never
    applicable are reported as ``not_applicable`` and diseases whose agents
    failed are reported as ``unavailable`` — neither is silently dropped.
    """
    disease_results = workflow.get("disease_results") or {}
    assessments: Dict[str, Any] = {}

    for disease in _known_diseases():
        entry = disease_results.get(disease)
        if not isinstance(entry, dict):
            assessments[disease] = {
                "status": STATUS_NOT_APPLICABLE,
                "risk_level": "Unavailable",
                "score": None,
                "factors": [],
                "reason": "No required parameters were submitted for this condition.",
            }
            continue

        analysis = entry.get("analysis") or {}
        risk = entry.get("risk") or {}
        final = entry.get("final_assessment") or {}
        factors = list(analysis.get("abnormal_parameters") or [])
        factors.extend(risk.get("triggers") or [])
        agent_statuses = entry.get("agent_status") or {}

        if final:
            status = final.get("status", STATUS_COMPLETED)
        elif risk:
            status = STATUS_COMPLETED_WITH_ERRORS
        else:
            status = STATUS_UNAVAILABLE

        assessments[disease] = {
            "status": status,
            "risk_level": final.get("risk_level") or risk.get("risk_level") or "Unavailable",
            "score": final.get("score", risk.get("risk_score")),
            "factors": list(dict.fromkeys(str(item) for item in factors)),
            "agent_status": agent_statuses,
            "analysis": analysis,
            "risk": risk,
            "symptoms": entry.get("symptom"),
            "trend": entry.get("trend"),
            "final_assessment": final,
        }

        # How much the score should be trusted, and which model produced it.
        # Only present when the disease was scored by a model that reports it.
        for key in ("model_confidence", "model_variant"):
            if risk.get(key) is not None:
                assessments[disease][key] = risk[key]

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
    differential_status = (
        (workflow.get("agent_status") or {}).get("differential", {}).get("status")
    )
    if not isinstance(differential, dict) or not differential:
        differential = {
            "status": differential_status or STATUS_UNAVAILABLE,
            "results": [],
        }
    else:
        differential = {"status": STATUS_COMPLETED, "results": differential}

    trend = workflow.get("trend_result")
    if not isinstance(trend, dict):
        trend = {"status": STATUS_UNAVAILABLE, "results": []}
    elif trend.get("status") not in {STATUS_COMPLETED, "insufficient_history"}:
        trend = {"status": STATUS_FAILED, "results": []}

    return {
        "assessment_id": assessment_id,
        "status": workflow.get("workflow_status", "unknown"),
        "assessments": assessments,
        "differential": differential,
        "trend": trend,
        "alerts": alerts,
        "explanation": workflow.get("explanation") or {},
        "agent_status": workflow.get("agent_status") or {},
        "created_at": created_at or workflow.get("timestamp"),
        "model_disclaimer": "These are model predictions and risk signals, not confirmed medical diagnoses.",
    }


def _known_diseases() -> List[str]:
    """Names of every registered disease plugin, with the workflow's own
    disease results appended so unregistered entries are still surfaced."""
    from Agents.plugins import registry

    names = list(registry.list_all())
    return names
