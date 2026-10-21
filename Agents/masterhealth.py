"""
Master Health Agent — LangGraph Orchestrator
Routes health analysis tasks to sub-agents via a compiled LangGraph StateGraph.

Workflow:
    START ─→ [extract_pdf] ─→ analyze ─→ [predict_risk] ─→ [check_symptoms]
         ─→ [merge_results] ─→ [generate_alert] ─→ finalize ─→ END

Nodes:
  extract_pdf   – Extracts health data from a PDF lab report (skipped for dict input)
  analyze       – Validates parameters & detects abnormalities (ReportAnalyzerAgent)
  predict_risk  – Predicts diabetes/metabolic risk (LLM → rule-based fallback)
    check_symptoms– Interprets symptom patterns (Symptom Checker Agent)
    merge_results – Combines lab + ML + symptom signals into final score/risk
    generate_alert– Creates patient-friendly alert/report (AlertAgent)
  finalize      – Marks the workflow as completed
"""

import logging
import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict

# Import sub-agents
from Agents.plugins.registry import applicable, register
from Agents.plugins.diabetes import DiabetesPlugin
from Agents.plugins.cardiovascular import CardiovascularPlugin
from Agents.plugins.boot import init_plugins
from Agents.trendagent import TrendAgent
from Agents.differentialagent import DifferentialAgent
from Agents.explainer import Explainer
from Agents.reportanalyzer import ReportAnalyzerAgent
from Agents.alertsystem import AlertAgent
from utils.helpers import normalize_level

# Initialize the disease plugins
init_plugins()

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  WORKFLOW STATE
# ══════════════════════════════════════════════════════════════════════════════

class HealthWorkflowState(TypedDict):
    """Typed state that flows through the LangGraph workflow."""
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
    # New fields for multi-disease support
    disease_results: Dict[str, Dict[str, Any]]
    agent_status: Dict[str, Dict[str, Any]]
    history: Optional[List[Dict]]
    trend_result: Optional[Dict]
    differential_result: Optional[Dict]
    explanation: Optional[str]


def _agent_status(status: str, error: Optional[str] = None) -> Dict[str, Any]:
    """Build the common status shape used by graph steps and disease agents."""
    result: Dict[str, Any] = {"status": status}
    if error:
        result["error"] = error
    return result


def _set_disease_status(
    disease_results: Dict[str, Dict[str, Any]],
    disease: str,
    agent: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    """Record an agent outcome without replacing that disease's other outputs."""
    entry = disease_results.setdefault(disease, {})
    statuses = entry.setdefault("agent_status", {})
    statuses[agent] = _agent_status(status, error)


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH NODES
# ══════════════════════════════════════════════════════════════════════════════

def extract_pdf_node(state: HealthWorkflowState) -> dict:
    """Extract health parameters from a PDF lab report."""
    try:
        from tools.labparse import extract_health_data_from_pdf

        pdf_path = state["pdf_path"]
        use_llm = state.get("use_llm", False)
        logger.info(f"NODE extract_pdf: {Path(pdf_path).name}")

        health_data = extract_health_data_from_pdf(pdf_path, use_llm=use_llm)
        filtered = {k: v for k, v in health_data.items() if v is not None}

        if not filtered:
            return {
                "workflow_status": "failed",
                "error": "No health parameters extracted from PDF",
                "steps": {**state["steps"], "0_pdf_extraction": {
                    "status": "failed", "error": "No parameters found"
                }},
            }

        return {
            "health_data": filtered,
            "steps": {**state["steps"], "0_pdf_extraction": {
                "status": "completed",
                "extracted_parameters": list(filtered.keys()),
                "extraction_method": "llm" if use_llm else "regex",
            }},
        }
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return {
            "workflow_status": "failed",
            "error": str(e),
            "steps": {**state["steps"], "0_pdf_extraction": {
                "status": "failed", "error": str(e)
            }},
        }


def analyze_report_node(state: HealthWorkflowState) -> dict:
    """Validate parameters & detect abnormalities per applicable disease."""
    try:
        logger.info("NODE analyze: Multi-disease ReportAnalyzer")
        from Agents.plugins.registry import applicable

        health_data = state["health_data"]
        plugins = applicable(health_data)

        if not plugins:
            return {
                "workflow_status": "failed",
                "error": "No applicable disease plugins found for the provided health data",
                "steps": {**state["steps"], "1_parameter_validation": {"status": "failed", "error": "No applicable plugins"}}
            }

        analyzer = ReportAnalyzerAgent()
        disease_results = dict(state.get("disease_results") or {})

        failed_diseases = []
        for plugin in plugins:
            try:
                analysis_json = analyzer.analyze_health_record(health_data, plugin=plugin)
                analysis = json.loads(analysis_json)
                if analysis.get("status") == "failed" or analysis.get("error"):
                    raise RuntimeError(
                        analysis.get("error", "unknown analyzer error")
                    )
                disease_results.setdefault(plugin.name, {})["analysis"] = analysis
                _set_disease_status(disease_results, plugin.name, "analysis", "completed")
            except Exception as exc:
                failed_diseases.append(plugin.name)
                logger.error("Analysis failed for %s: %s", plugin.name, exc)
                _set_disease_status(
                    disease_results, plugin.name, "analysis", "failed", str(exc)
                )

        successful_plugins = [
            plugin for plugin in plugins
            if disease_results.get(plugin.name, {}).get("analysis")
        ]
        if not successful_plugins:
            return {
                "workflow_status": "failed",
                "error": "All applicable disease analyses failed",
                "disease_results": disease_results,
                "steps": {
                    **state["steps"],
                    "1_parameter_validation": {
                        "status": "failed",
                        "error": "All applicable disease analyses failed",
                        "failed_diseases": failed_diseases,
                    },
                },
            }

        # Primary is the first successfully analyzed applicable plugin.
        primary_name = successful_plugins[0].name
        primary_analysis = disease_results[primary_name]["analysis"]

        updated_steps = {**state["steps"]}
        updated_steps["1_parameter_validation"] = {
            "status": "completed_with_errors" if failed_diseases else "completed",
            "validated_parameters": list(primary_analysis.get("parameters", {}).keys()),
            "applicable_diseases": [p.name for p in plugins]
        }
        updated_steps["2_abnormality_detection"] = {
            "status": "completed",
            "abnormal_count": primary_analysis.get("abnormal_count", 0),
            "abnormal_parameters": primary_analysis.get("abnormal_parameters", []),
            "failed_diseases": failed_diseases,
        }
        updated_steps["3_structured_output"] = {
            "status": "completed",
            "format": "JSON",
            "analysis_result": primary_analysis,
        }

        return {
            "analysis_result": primary_analysis,
            "disease_results": disease_results,
            "agent_status": {
                name: entry.get("agent_status", {}).get("analysis", _agent_status("failed"))
                for name, entry in disease_results.items()
            },
            "steps": updated_steps
        }
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {"workflow_status": "failed", "error": str(e)}


def predict_risk_node(state: HealthWorkflowState) -> dict:
    """Predict health risk per applicable disease."""
    try:
        logger.info("NODE predict_risk: Multi-disease")
        from Agents.plugins.registry import applicable

        health_data = state.get("health_data", {})
        disease_results = dict(state.get("disease_results") or {})
        plugins = applicable(health_data)

        disease_risks = {}
        failed_diseases = []

        for plugin in plugins:
            name = plugin.name
            analysis = (disease_results.get(name) or {}).get("analysis", {})

            if not analysis:
                failed_diseases.append(name)
                _set_disease_status(
                    disease_results, name, "risk", "skipped", "Analysis unavailable"
                )
                continue

            try:
                # All plugins receive the same flat input plus their own analysis.
                # LLM-backed prediction is opt-in; the default is deterministic.
                if name == "diabetes" and not state.get("use_llm", False):
                    risk = plugin.rule_risk(health_data, analysis)
                else:
                    risk = plugin.predict_risk(health_data, analysis)
                if not isinstance(risk, dict) or risk.get("error"):
                    raise RuntimeError(
                        risk.get("error", "Risk predictor returned no result")
                        if isinstance(risk, dict)
                        else "Risk predictor returned an invalid result"
                    )
                risk["risk_level"] = normalize_level(risk.get("risk_level", "Low"))
                disease_risks[name] = risk
                disease_results.setdefault(name, {})["risk"] = risk
                status = "completed_with_errors" if risk.get("fallback_reason") else "completed"
                _set_disease_status(disease_results, name, "risk", status, risk.get("fallback_reason"))
            except Exception as exc:
                failed_diseases.append(name)
                logger.error("Risk prediction failed for %s: %s", name, exc)
                _set_disease_status(disease_results, name, "risk", "failed", str(exc))

        if not disease_risks:
            return {
                "workflow_status": "failed",
                "error": "No risk predictions generated",
                "disease_results": disease_results,
                "agent_status": {"risk": _agent_status("failed", "No risk predictions generated")},
                "steps": {
                    **state["steps"],
                    "4_risk_prediction": {
                        "status": "failed",
                        "error": "No risk predictions generated",
                        "failed_diseases": failed_diseases,
                    },
                },
            }

        primary_name = next(
            plugin.name for plugin in plugins if plugin.name in disease_risks
        )
        primary_risk = disease_risks[primary_name]

        updated_steps = {**state["steps"]}
        updated_steps["4_risk_prediction"] = {
            "status": "completed_with_errors" if failed_diseases else "completed",
            "risk_level": primary_risk.get("risk_level"),
            "risk_probability": primary_risk.get("risk_probability"),
            "disease_risks": disease_risks,
            "failed_diseases": failed_diseases,
        }

        return {
            "risk_result": primary_risk,
            "disease_results": disease_results,
            "agent_status": {"risk": _agent_status(
                "completed_with_errors" if failed_diseases else "completed"
            )},
            "steps": updated_steps
        }
    except Exception as e:
        logger.error(f"Risk prediction failed: {e}")
        return {
            "risk_result": {"error": str(e)},
            "steps": {**state["steps"], "4_risk_prediction": {"status": "failed", "error": str(e)}},
        }


def generate_alert_node(state: HealthWorkflowState) -> dict:
    """Generate final patient-facing alert/report from winning disease and analysis + risk outputs."""
    try:
        logger.info("NODE generate_alert")

        final_assessment = state.get("final_assessment") or {}
        winning_disease = final_assessment.get("disease", "diabetes")

        alert_agent = AlertAgent(use_llm=state.get("use_llm", False))
        payload = {
            "disease": winning_disease,
            "health_data": state.get("health_data", {}),
            "analysis_result": state.get("analysis_result", {}),
            "risk_result": state.get("risk_result", {}),
            "risk_level": normalize_level(final_assessment.get("risk_level", "Low")),
            "probability": final_assessment.get("score"),
            "diabetes_probability": final_assessment.get("score"),
            "abnormal_parameters": (state.get("analysis_result") or {}).get("abnormal_count", 0),
            "symptom_score": final_assessment.get("symptom_score"),
            "symptoms": state.get("symptoms") or [],
            "symptom_result": state.get("symptom_result") or {},
            "manual_text": state.get("manual_text"),
            "explanation": state.get("explanation"),
            "final_assessment": final_assessment,
        }
        alert_result = alert_agent.process(payload)

        return {
            "alert_result": alert_result,
            "steps": {**state["steps"], "7_alert_generation": {
                "status": "completed",
                "alert": alert_result.get("alert", False),
                "risk_level": normalize_level(alert_result.get("risk_level", "Unknown")),
                "alert_result": alert_result,
            }},
        }
    except Exception as e:
        logger.error(f"Alert generation failed: {e}")
        return {
            "alert_result": {"error": str(e)},
            "steps": {**state["steps"], "7_alert_generation": {
                "status": "failed", "error": str(e)
            }},
        }


def check_symptoms_node(state: HealthWorkflowState) -> dict:
    """Run symptom checker per applicable disease."""
    try:
        logger.info("NODE check_symptoms: Multi-disease")
        from Agents.plugins.registry import applicable

        symptoms = state.get("symptoms") or []
        manual_text = state.get("manual_text")
        health_data = state.get("health_data", {})
        plugins = applicable(health_data)
        disease_results = dict(state.get("disease_results") or {})

        disease_symptoms = {}

        if not symptoms and not manual_text:
            default_sym = {
                "symptom_alignment": "low",
                "severity_score": 0.0,
                "reasoning": "No symptoms provided; symptom contribution kept minimal.",
                "input_summary": {
                    "symptoms_count": 0,
                    "sources_used": [],
                    "top_hypothesis": "No symptoms provided",
                },
                "model_used": "rule-based",
                "symptom_mapping": {
                    "condition_hypotheses": [],
                    "top_hypothesis": "No symptoms provided",
                    "total_symptoms_checked": 0,
                    "unmatched_symptoms": [],
                },
            }
            for plugin in plugins:
                disease_symptoms[plugin.name] = default_sym
                if plugin.name not in disease_results:
                    disease_results[plugin.name] = {}
                disease_results[plugin.name]["symptom"] = default_sym
                _set_disease_status(disease_results, plugin.name, "symptom", "completed")
            primary_name = plugins[0].name if plugins else "diabetes"
            symptom_result = default_sym
        else:
            for plugin in plugins:
                name = plugin.name
                if name == "diabetes":
                    try:
                        from Agents.symptomchecker import SymptomCheckerAgent
                        checker = SymptomCheckerAgent()
                        report_ctx = (disease_results.get(name) or {}).get("analysis") or state.get("analysis_result")
                        res = checker.analyze_symptoms(
                            symptoms=symptoms,
                            report_context=report_ctx,
                            manual_text=manual_text,
                            use_llm=state.get("use_llm", False),
                        )
                        res.update(_derive_symptom_metrics(res))
                        disease_symptoms[name] = res
                    except Exception as e:
                        logger.error(f"Diabetes symptom check failed: {e}")
                        disease_symptoms[name] = {
                            "error": str(e),
                            "severity_score": 0.0,
                            "symptom_alignment": "low",
                        }
                        _set_disease_status(disease_results, name, "symptom", "failed", str(e))
                else:
                    try:
                        disease_symptoms[name] = plugin.score_symptoms(symptoms)
                        _set_disease_status(disease_results, name, "symptom", "completed")
                    except Exception as exc:
                        logger.error("Symptom scoring failed for %s: %s", name, exc)
                        disease_symptoms[name] = {"error": str(exc), "severity_score": 0.0}
                        _set_disease_status(disease_results, name, "symptom", "failed", str(exc))

                if name not in disease_results:
                    disease_results[name] = {}
                disease_results[name]["symptom"] = disease_symptoms[name]

            if not disease_symptoms:
                raise RuntimeError("No disease plugins applicable for symptom check")

            primary_name = plugins[0].name
            symptom_result = disease_symptoms[primary_name]

        return {
            "symptom_result": symptom_result,
            "disease_results": disease_results,
            "agent_status": {"symptom": _agent_status("completed")},
            "steps": {**state["steps"], "5_symptom_checker": {
                "status": "completed",
                "symptom_alignment": symptom_result.get("symptom_alignment"),
                "severity_score": symptom_result.get("severity_score"),
                "top_hypothesis": (symptom_result.get("input_summary") or {}).get("top_hypothesis"),
                "symptom_result": symptom_result,
            }},
        }
    except Exception as e:
        logger.error(f"Symptom checker failed: {e}")
        return {
            "symptom_result": {
                "symptom_alignment": "low",
                "severity_score": 0.0,
                "error": str(e),
            },
            "agent_status": {"symptom": _agent_status("failed", str(e))},
            "steps": {**state["steps"], "5_symptom_checker": {
                "status": "failed", "error": str(e)
            }},
        }


def merge_results_node(state: HealthWorkflowState) -> dict:
    """Combine signals per disease, fuse with individual trend, and find the highest risk."""
    try:
        logger.info("NODE merge_results: Multi-disease Fusion")

        health_data = state.get("health_data", {})
        from Agents.plugins.registry import applicable
        plugins = applicable(health_data)

        disease_results = dict(state.get("disease_results") or {})

        all_fusions = {}
        for plugin in plugins:
            name = plugin.name
            entry = disease_results.get(name, {})
            plugin_trend = entry.get("trend") or state.get("trend_result")
            fusion = _fuse(
                analysis=entry.get("analysis", {}),
                risk=entry.get("risk", {}),
                symptom=entry.get("symptom", {}),
                trend=plugin_trend
            )
            if fusion.get("score") is not None:
                fusion["risk_level"] = normalize_level(fusion.get("risk_level", "Low"))
            all_fusions[name] = fusion
            disease_results.setdefault(name, {})["final_assessment"] = fusion

        valid_fusions = {
            name: fusion for name, fusion in all_fusions.items()
            if fusion.get("score") is not None
        }
        if not valid_fusions:
            return {"workflow_status": "failed", "error": "No disease results to merge"}

        # Winning disease is the highest scoring disease
        winning_name = max(valid_fusions, key=lambda k: valid_fusions[k]["score"])
        winning_assessment = valid_fusions[winning_name]
        winning_entry = disease_results.get(winning_name, {})

        updated_steps = {**state["steps"]}
        updated_steps["6_result_fusion"] = {
            "status": (
                "completed_with_errors"
                if len(valid_fusions) != len(all_fusions)
                else "completed"
            ),
            "risk_level": winning_assessment["risk_level"],
            "score": winning_assessment["score"],
            "winning_disease": winning_name,
            "final_assessment": winning_assessment,
            "all_diseases": all_fusions,
            "unavailable_diseases": [
                name for name in all_fusions if name not in valid_fusions
            ],
        }

        # Rebuild analysis_result, risk_result, symptom_result from the winning disease
        return {
            "analysis_result": winning_entry.get("analysis") or state.get("analysis_result"),
            "risk_result": winning_entry.get("risk") or state.get("risk_result"),
            "symptom_result": winning_entry.get("symptom") or state.get("symptom_result"),
            "final_assessment": {
                **winning_assessment,
                "disease": winning_name,
                "all_diseases": all_fusions,
            },
            "disease_results": disease_results,
            "agent_status": {"fusion": _agent_status(
                "completed_with_errors"
                if len(valid_fusions) != len(all_fusions)
                else "completed"
            )},
            "steps": updated_steps
        }
    except Exception as e:
        logger.error(f"Result fusion failed: {e}")
        return {
            "final_assessment": {"error": str(e)},
            "steps": {**state["steps"], "6_result_fusion": {
                "status": "failed", "error": str(e)
            }},
        }


def finalize_node(state: HealthWorkflowState) -> dict:
    """Mark the workflow as completed (or completed_with_errors if errors occurred)."""
    if state.get("workflow_status") == "failed":
        return {}
    steps = state.get("steps") or {}
    has_errors = any(
        isinstance(s, dict) and s.get("status") == "failed" for s in steps.values()
    ) or (state.get("final_assessment") or {}).get("error") is not None

    if has_errors:
        return {"workflow_status": "completed_with_errors"}
    return {"workflow_status": "completed"}
    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  RISK PREDICTION FALLBACK (RULE-BASED ONLY)
# ══════════════════════════════════════════════════════════════════════════════

def _llm_risk_prediction(analysis: Dict, health_data: Dict) -> Dict[str, Any]:
    """Fallback risk prediction helper.

    Originally this used an LLM (via Ollama) and fell back to
    deterministic rules on failure. To remove the external dependency
    and deprecation warnings, this implementation now delegates
    directly to the rule-based engine.
    """
    logger.info("LLM-based risk prediction disabled; using rule-based implementation.")
    return _rule_based_risk(analysis, health_data)


def _rule_based_risk(analysis: Dict, health_data: Dict) -> Dict[str, Any]:
    """Deterministic rule-based risk scoring (always available).

    Delegates to the shared compute_diabetes_rule_risk function so
    _rule_based_risk, _compute_lab_risk, and DiabetesPlugin.rule_risk()
    all execute the single unified clinical rule scorer.
    """
    from Agents.riskpredictor import compute_diabetes_rule_risk
    res = compute_diabetes_rule_risk(health_data, analysis)

    raw_level = res.get("risk_level") or "Low"
    risk_level = normalize_level(raw_level)
    risk_score = res.get("risk_score") or 0
    risk_probability = res.get("risk_probability")
    if risk_probability is None:
        risk_probability = round(risk_score / 100.0, 2)

    risk_factors = []
    recommendations = []
    hba1c = health_data.get("hba1c", 0)
    glucose = health_data.get("glucose", 0)
    bmi = health_data.get("bmi", 0)
    age = health_data.get("age", 0)

    if hba1c >= 6.5:
        risk_factors.append({"parameter": "hba1c", "value": hba1c, "impact": "High"})
        recommendations.append("Consult endocrinologist for diabetes management")
    elif hba1c >= 5.7:
        risk_factors.append({"parameter": "hba1c", "value": hba1c, "impact": "Medium"})
        recommendations.append("Monitor HbA1c every 3 months")

    if glucose >= 126:
        risk_factors.append({"parameter": "glucose", "value": glucose, "impact": "High"})
        recommendations.append("Regular fasting glucose monitoring recommended")
    elif glucose >= 100:
        risk_factors.append({"parameter": "glucose", "value": glucose, "impact": "Medium"})

    if bmi >= 25:
        risk_factors.append({"parameter": "bmi", "value": bmi, "impact": "High" if bmi >= 30 else "Medium"})
        recommendations.append("Weight management program recommended")

    if age >= 45:
        risk_factors.append({"parameter": "age", "value": age, "impact": "Medium"})

    if not recommendations:
        recommendations.append("Maintain healthy lifestyle and regular checkups")

    return {
        "risk_level": risk_level,
        "risk_probability": risk_probability,
        "risk_percentage": f"{risk_score}%",
        "prediction_method": "rule_based",
        "risk_score": risk_score,
        "risk_factors": risk_factors,
        "recommendations": recommendations,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TREND, DIFFERENTIAL, & EXPLANATION NODES
# ══════════════════════════════════════════════════════════════════════════════

def trend_analysis_node(state: HealthWorkflowState) -> dict:
    """Perform trend analysis per applicable disease on historical health records."""
    try:
        logger.info("NODE trend: Analyzing historical trends per disease")
        from Agents.plugins.registry import applicable

        health_data = state["health_data"]
        plugins = applicable(health_data)
        history = state.get("history") or []
        disease_results = dict(state.get("disease_results") or {})

        if not plugins:
            res = {"status": "no_plugins", "trend_score": 0.0, "direction": "stable", "visits_used": 0}
            return {"trend_result": res, "steps": {**state["steps"], "5b_trend_analysis": {"status": "completed", "result": res}}}

        trends_per_plugin = {}
        for plugin in plugins:
            try:
                t_res = TrendAgent.run_trend(history, plugin, health_data)
            except Exception as te:
                logger.warning(f"Trend analysis failed for {plugin.name}: {te}")
                t_res = {"trend_score": 0.0, "direction": "stable", "visits_used": 0, "status": "failed", "error": str(te)}
            trends_per_plugin[plugin.name] = t_res
            if plugin.name not in disease_results:
                disease_results[plugin.name] = {}
            disease_results[plugin.name]["trend"] = t_res

        primary_name = plugins[0].name
        primary_trend = trends_per_plugin.get(primary_name) or {
            "trend_score": 0.0, "direction": "stable", "visits_used": 0, "status": "insufficient_history"
        }

        return {
            "trend_result": primary_trend,
            "disease_results": disease_results,
            "agent_status": {"trend": _agent_status(
                "completed_with_errors"
                if any(result.get("status") == "failed" for result in trends_per_plugin.values())
                else "completed"
            )},
            "steps": {**state["steps"], "5b_trend_analysis": {
                "status": (
                    "completed_with_errors"
                    if any(result.get("status") == "failed" for result in trends_per_plugin.values())
                    else "completed"
                ),
                "result": primary_trend,
                "all_trends": trends_per_plugin
            }}
        }
    except Exception as e:
        logger.warning(f"Trend analysis failed gracefully: {e}")
        fallback_trend = {"trend_score": 0.0, "direction": "stable", "visits_used": 0, "status": "failed", "error": str(e)}
        return {
            "trend_result": fallback_trend,
            "steps": {**state["steps"], "5b_trend_analysis": {"status": "completed", "result": fallback_trend, "warning": str(e)}}
        }


def differential_analysis_node(state: HealthWorkflowState) -> dict:
    """Perform differential diagnosis analysis across suspected diseases."""
    try:
        logger.info("NODE differential: Performing differential analysis")
        disease_results = state.get("disease_results", {})
        symptoms = state.get("symptoms") or []

        result = DifferentialAgent.run_differential(disease_results, symptoms)

        return {
            "differential_result": result,
            "agent_status": {"differential": _agent_status(
                "completed" if result is not None else "skipped"
            )},
            "steps": {**state["steps"], "5c_differential_analysis": {
                "status": "completed" if result is not None else "skipped",
                "result": result
            }}
        }
    except Exception as e:
        logger.warning(f"Differential analysis failed gracefully: {e}")
        return {
            "differential_result": None,
            "agent_status": {"differential": _agent_status("failed", str(e))},
            "steps": {**state["steps"], "5c_differential_analysis": {
                "status": "failed",
                "result": None,
                "error": str(e)
            }}
        }


def explanation_node(state: HealthWorkflowState) -> dict:
    """Generate medical explanation for the final assessment."""
    try:
        logger.info("NODE explain: Generating clinical explanation")
        final_assessment = state.get("final_assessment")
        trend = state.get("trend_result")

        if not final_assessment:
            return {
                "explanation": "No assessment available to explain.",
                "steps": {**state["steps"], "6b_explanation_generation": {"status": "completed", "explanation": "No assessment available."}}
            }

        triggers = final_assessment.get("triggers") or []
        explanation = Explainer.explain(final_assessment, trend, triggers)

        return {
            "explanation": explanation,
            "steps": {**state["steps"], "6b_explanation_generation": {
                "status": "completed",
                "explanation": explanation
            }}
        }
    except Exception as e:
        logger.warning(f"Explanation generation failed gracefully: {e}")
        return {
            "explanation": "Clinical risk assessment completed.",
            "steps": {**state["steps"], "6b_explanation_generation": {
                "status": "completed",
                "warning": str(e)
            }}
        }


# ══════════════════════════════════════════════════════════════════════════════
#  CONDITIONAL ROUTING
# ══════════════════════════════════════════════════════════════════════════════

def _route_entry(state: HealthWorkflowState) -> str:
    """START → extract_pdf (PDF input) or analyze (dict input)."""
    if state.get("input_type") == "pdf":
        return "extract_pdf"
    return "analyze"


def _route_after_pdf(state: HealthWorkflowState) -> str:
    """After PDF extraction: continue to analyze or abort on failure."""
    if state.get("workflow_status") == "failed":
        return "finalize"
    return "analyze"


def _route_after_analysis(state: HealthWorkflowState) -> str:
    """After analysis: route to risk/symptom/merge/finalize based on config."""
    if state.get("workflow_status") == "failed":
        return "finalize"
    if state.get("include_risk", True):
        return "predict_risk"
    if state.get("include_symptom", True):
        return "check_symptoms"
    return "trend"

def _route_after_risk(state: HealthWorkflowState) -> str:
    """After risk: route to symptom checker when enabled, else merge directly."""
    if state.get("workflow_status") == "failed":
        return "finalize"
    if state.get("include_symptom", True):
        return "check_symptoms"
    return "trend"

def _route_after_symptoms(state: HealthWorkflowState) -> str:
    """After symptoms: route to trend analysis."""
    if state.get("workflow_status") == "failed":
        return "finalize"
    return "trend"

def _route_after_trend(state: HealthWorkflowState) -> str:
    """After trend: route to differential agent."""
    if state.get("workflow_status") == "failed":
        return "finalize"
    return "differential"

def _route_after_differential(state: HealthWorkflowState) -> str:
    """After differential: route to result merge."""
    if state.get("workflow_status") == "failed":
        return "finalize"
    return "merge_results"

def _route_after_merge(state: HealthWorkflowState) -> str:
    """After merge: route to explanation."""
    if state.get("workflow_status") == "failed":
        return "finalize"
    return "explain"

def _route_after_explain(state: HealthWorkflowState) -> str:
    """After explain: send result to alert agent or finish."""
    if state.get("workflow_status") == "failed":
        return "finalize"
    if state.get("include_alert", True):
        return "generate_alert"
    return "finalize"


def _to_probability(value: Any) -> Optional[float]:
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


def _score_to_risk_level(score: float) -> str:
    """Map numeric score to standardized risk level."""
    if score >= 0.80:
        return "Critical"
    elif score >= 0.65:
        return "High"
    elif score >= 0.40:
        return "Moderate"
    else:
        return "Low"


def _fuse(analysis: Dict, risk: Dict, symptom: Dict, trend: Optional[Dict] = None) -> Dict:
    """
    Combine ML, abnormal labs, symptom severity, and trend trajectory into a final score.
    Implements the Major Project Extension formula.
    """
    component_status = {
        "analysis": "completed" if analysis and not analysis.get("error") else "unavailable",
        "risk": "completed" if risk and not risk.get("error") else "unavailable",
        "symptom": "completed" if symptom and not symptom.get("error") else "unavailable",
        "trend": "completed" if trend and trend.get("status") == "completed" else "unavailable",
    }
    if component_status["analysis"] == "unavailable" or component_status["risk"] == "unavailable":
        return {
            "status": "unavailable",
            "risk_level": "Unavailable",
            "score": None,
            "component_status": component_status,
            "reason": "Required analysis or risk result is unavailable.",
        }

    abnormal_count = int(analysis.get("abnormal_count", 0) or 0)
    abnormal_weight = min(max(abnormal_count, 0) / 3.0, 1.0)

    ml_probability = _to_probability(
        risk.get("risk_probability")
        or risk.get("risk_percentage")
        or ((risk.get("score") / 100.0) if risk.get("score") is not None else None)
    )
    if ml_probability is None:
        ml_probability = 0.0

    symptom_score = symptom.get("severity_score")
    if symptom_score is None:
        symptom_score = symptom.get("severity", 0.0)
    try:
        symptom_score = float(symptom_score)
    except (TypeError, ValueError):
        symptom_score = 0.0
    symptom_score = max(0.0, min(1.0, symptom_score))

    # Trend Component
    trend_score = 0.0
    if component_status["trend"] == "completed":
        trend_score = trend.get("trend_score", 0.0)

    # Determine which formula to use based on history (at least 2 visits total: 1 past + 1 current)
    has_history = (trend and trend.get("visits_used", 0) >= 2)

    if has_history:
        # Major Project Formula: 0.50*risk + 0.25*abnormal + 0.10*symptom + 0.15*trend
        weights = {"ml": 0.50, "abnormal": 0.25, "symptom": 0.10, "trend": 0.15}
        formula = "0.50*risk + 0.25*abnormal + 0.10*symptom + 0.15*trend"
    else:
        # Fallback Formula: 0.60*risk + 0.30*abnormal + 0.10*symptom
        weights = {"ml": 0.60, "abnormal": 0.30, "symptom": 0.10, "trend": 0.0}
        formula = "0.60*risk + 0.30*abnormal + 0.10*symptom"

    components = {
        "ml": ml_probability,
        "abnormal": abnormal_weight,
        "symptom": symptom_score,
        "trend": trend_score,
    }
    available_weights = {
        name: weight for name, weight in weights.items()
        if name == "ml" or name == "abnormal"
        or (name == "symptom" and component_status["symptom"] == "completed")
        or (name == "trend" and component_status["trend"] == "completed")
    }
    weight_total = sum(available_weights.values()) or 1.0
    final_score = round(
        sum(components[name] * weight for name, weight in available_weights.items())
        / weight_total,
        4,
    )

    final_risk = normalize_level(_score_to_risk_level(final_score))

    return {
        "risk_level": final_risk,
        "status": "completed" if len(available_weights) == len(weights) else "partial",
        "score": final_score,
        "ml_probability": ml_probability,
        "abnormal_count": abnormal_count,
        "abnormal_count_weight": round(abnormal_weight, 4),
        "symptom_score": round(symptom_score, 4),
        "trend_score": round(trend_score, 4),
        "formula": formula,
        "component_status": component_status,
        "components": {
            "ml_component": round(ml_probability * weights["ml"], 4),
            "abnormal_component": round(abnormal_weight * weights["abnormal"], 4),
            "symptom_component": round(symptom_score * weights["symptom"], 4),
            "trend_component": round(trend_score * weights["trend"], 4),
        },
    }



def _derive_symptom_metrics(symptom_result: Dict[str, Any]) -> Dict[str, Any]:
    """Derive normalized severity/alignment from symptom mapping output."""
    mapping = symptom_result.get("symptom_mapping") or {}
    hyps = mapping.get("condition_hypotheses") or []
    total = int(mapping.get("total_symptoms_checked") or 0)

    if not hyps or total <= 0:
        severity = 0.0
    else:
        top_match_count = int(hyps[0].get("match_count") or 0)
        severity = max(0.0, min(1.0, top_match_count / max(total, 1)))

    if severity >= 0.7:
        alignment = "high"
    elif severity >= 0.35:
        alignment = "moderate"
    else:
        alignment = "low"

    return {
        "symptom_alignment": alignment,
        "severity_score": round(severity, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD & COMPILE LANGGRAPH WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════

def build_health_workflow():
    """Construct and compile the health-analysis StateGraph."""
    graph = StateGraph(HealthWorkflowState)

    # -- nodes --
    graph.add_node("extract_pdf", extract_pdf_node)
    graph.add_node("analyze", analyze_report_node)
    graph.add_node("predict_risk", predict_risk_node)
    graph.add_node("check_symptoms", check_symptoms_node)
    graph.add_node("trend", trend_analysis_node)
    graph.add_node("differential", differential_analysis_node)
    graph.add_node("merge_results", merge_results_node)
    graph.add_node("explain", explanation_node)
    graph.add_node("generate_alert", generate_alert_node)
    graph.add_node("finalize", finalize_node)

    # -- edges --
    graph.add_conditional_edges(START, _route_entry, {
        "extract_pdf": "extract_pdf",
        "analyze": "analyze",
    })
    graph.add_conditional_edges("extract_pdf", _route_after_pdf, {
        "finalize": "finalize",
        "analyze": "analyze",
    })
    graph.add_conditional_edges("analyze", _route_after_analysis, {
        "predict_risk": "predict_risk",
        "check_symptoms": "check_symptoms",
        "trend": "trend",
        "finalize": "finalize",
    })
    graph.add_conditional_edges("predict_risk", _route_after_risk, {
        "check_symptoms": "check_symptoms",
        "trend": "trend",
        "finalize": "finalize",
    })
    graph.add_conditional_edges("check_symptoms", _route_after_symptoms, {
        "trend": "trend",
        "finalize": "finalize",
    })
    graph.add_conditional_edges("trend", _route_after_trend, {
        "differential": "differential",
        "finalize": "finalize",
    })
    graph.add_conditional_edges("differential", _route_after_differential, {
        "merge_results": "merge_results",
        "finalize": "finalize",
    })
    graph.add_conditional_edges("merge_results", _route_after_merge, {
        "explain": "explain",
        "finalize": "finalize",
    })
    graph.add_conditional_edges("explain", _route_after_explain, {
        "generate_alert": "generate_alert",
        "finalize": "finalize",
    })
    graph.add_edge("generate_alert", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER HEALTH AGENT (public API — same interface as before)
# ══════════════════════════════════════════════════════════════════════════════

class MasterHealthAgent:
    """
    Orchestrates health analysis through a compiled LangGraph workflow.

    Public methods keep the same signature so that healthroutes.py and
    other consumers continue to work without changes.
    """

    def __init__(self):
        logger.info("Initializing Master Health Agent (LangGraph)...")
        self.workflow = build_health_workflow()
        logger.info("Master Health Agent ready — workflow compiled")

    # ── process dict input ────────────────────────────────────────────────

    def process_health_data(self, health_data: Dict[str, Any],
                            symptoms: Optional[List[str]] = None,
                            manual_text: Optional[str] = None,
                            use_llm: bool = False,
                            include_risk: bool = True,
                            include_symptom: bool = True,
                            include_alert: bool = True,
                            history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Process health data through the LangGraph workflow.

        Args:
            health_data: {"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45}
            symptoms: ["fatigue", "frequent urination"]
            include_risk: Whether to include risk prediction step
            history: Historical health records for trend analysis
        """
        payload = dict(health_data)
        extracted_symptoms = symptoms

        # Supports frontend payloads where symptoms are embedded in health_data.
        if extracted_symptoms is None and isinstance(payload.get("symptoms"), list):
            extracted_symptoms = payload.pop("symptoms")
        else:
            payload.pop("symptoms", None)

        initial_state: HealthWorkflowState = {
            "health_data": payload,
            "symptoms": extracted_symptoms or [],
            "manual_text": manual_text,
            "include_risk": include_risk,
            "include_symptom": include_symptom,
            "include_alert": include_alert,
            "input_type": "data",
            "pdf_path": None,
            "use_llm": use_llm,
            "analysis_result": None,
            "risk_result": None,
            "symptom_result": None,
            "final_assessment": None,
            "alert_result": None,
            "workflow_status": "in_progress",
            "error": None,
            "steps": {},
            "timestamp": str(datetime.now()),
            "input_file": None,
            "disease_results": {},
            "history": history,
            "trend_result": None,
            "differential_result": None,
            "explanation": None,
        }

        result = self.workflow.invoke(initial_state)

        return {
            "workflow_status": result["workflow_status"],
            "timestamp": result["timestamp"],
            "steps": result["steps"],
            **({"symptom_result": result["symptom_result"]} if result.get("symptom_result") else {}),
            **({"final_assessment": result["final_assessment"]} if result.get("final_assessment") else {}),
            **({"alert_result": result["alert_result"]} if result.get("alert_result") else {}),
            **({
                "risk_level": result["alert_result"].get("risk_level"),
                "alert": result["alert_result"].get("alert"),
                "report": result["alert_result"].get("report"),
            } if result.get("alert_result") else {}),
            **({"disease_results": result.get("disease_results")} if result.get("disease_results") else {}),
            **({"trend_result": result.get("trend_result")} if result.get("trend_result") else {}),
            **({"differential_result": result.get("differential_result")} if result.get("differential_result") else {}),
            **({"explanation": result.get("explanation")} if result.get("explanation") else {}),
            **({"error": result["error"]} if result.get("error") else {}),
        }

    # ── process PDF input ─────────────────────────────────────────────────

    def process_pdf_report(self, pdf_path: str,
                           use_llm: bool = False,
                           symptoms: Optional[List[str]] = None,
                           manual_text: Optional[str] = None,
                           include_risk: bool = True,
                           include_symptom: bool = True,
                           include_alert: bool = True,
                           history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Process a PDF lab report through the LangGraph workflow.

        Args:
            pdf_path: Path to the PDF file
            use_llm: Use LLM for PDF text extraction
            include_risk: Whether to include risk prediction step
            history: Historical health records for trend analysis
        """
        initial_state: HealthWorkflowState = {
            "health_data": {},
            "symptoms": symptoms or [],
            "manual_text": manual_text,
            "include_risk": include_risk,
            "include_symptom": include_symptom,
            "include_alert": include_alert,
            "input_type": "pdf",
            "pdf_path": pdf_path,
            "use_llm": use_llm,
            "analysis_result": None,
            "risk_result": None,
            "symptom_result": None,
            "final_assessment": None,
            "alert_result": None,
            "workflow_status": "in_progress",
            "error": None,
            "steps": {},
            "timestamp": str(datetime.now()),
            "input_file": Path(pdf_path).name,
            "disease_results": {},
            "history": history,
            "trend_result": None,
            "differential_result": None,
            "explanation": None,
        }

        result = self.workflow.invoke(initial_state)

        return {
            "workflow_status": result["workflow_status"],
            "timestamp": result["timestamp"],
            "input_type": "pdf",
            "input_file": result.get("input_file", Path(pdf_path).name),
            "steps": result["steps"],
            **({"symptom_result": result["symptom_result"]} if result.get("symptom_result") else {}),
            **({"final_assessment": result["final_assessment"]} if result.get("final_assessment") else {}),
            **({"alert_result": result["alert_result"]} if result.get("alert_result") else {}),
            **({
                "risk_level": result["alert_result"].get("risk_level"),
                "alert": result["alert_result"].get("alert"),
                "report": result["alert_result"].get("report"),
            } if result.get("alert_result") else {}),
            **({"disease_results": result.get("disease_results")} if result.get("disease_results") else {}),
            **({"trend_result": result.get("trend_result")} if result.get("trend_result") else {}),
            **({"differential_result": result.get("differential_result")} if result.get("differential_result") else {}),
            **({"explanation": result.get("explanation")} if result.get("explanation") else {}),
            **({"error": result["error"]} if result.get("error") else {}),
        }

    # ── human-readable report ────────────────────────────────────────────

    def get_complete_report(self, workflow_result: Dict[str, Any]) -> str:
        """Generate human-readable report from workflow result."""
        lines = []
        lines.append("=" * 70)
        lines.append("HEALTH ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {workflow_result.get('timestamp', 'N/A')}")
        lines.append(f"Status: {workflow_result.get('workflow_status', 'Unknown').upper()}")

        if workflow_result.get("input_type") == "pdf":
            lines.append(f"Input: PDF Report - {workflow_result.get('input_file', 'N/A')}")

        lines.append("")
        lines.append("WORKFLOW EXECUTION:")
        lines.append("-" * 70)

        steps = workflow_result.get("steps", {})

        # Step 0 — PDF Extraction
        if "0_pdf_extraction" in steps:
            step = steps["0_pdf_extraction"]
            icon = "✅" if step["status"] == "completed" else "❌"
            lines.append(f"{icon} Step 0: PDF Data Extraction")
            lines.append(f"   Method: {step.get('extraction_method', 'N/A')}")
            lines.append(f"   Parameters: {', '.join(step.get('extracted_parameters', []))}")
            lines.append("")

        # Step 1 — Parameter Validation
        if "1_parameter_validation" in steps:
            step = steps["1_parameter_validation"]
            icon = "✅" if step["status"] == "completed" else "❌"
            lines.append(f"{icon} Step 1: Parameter Validation")
            lines.append(f"   Validated: {', '.join(step.get('validated_parameters', []))}")
            lines.append("")

        # Step 2 — Abnormality Detection
        if "2_abnormality_detection" in steps:
            step = steps["2_abnormality_detection"]
            icon = "✅" if step["status"] == "completed" else "❌"
            lines.append(f"{icon} Step 2: Abnormality Detection")
            lines.append(f"   Abnormal Count: {step.get('abnormal_count', 0)}")
            if step.get("abnormal_parameters"):
                lines.append(f"   Abnormal Parameters: {', '.join(step['abnormal_parameters'])}")
            lines.append("")

        # Step 3 — Structured Output
        if "3_structured_output" in steps:
            step = steps["3_structured_output"]
            icon = "✅" if step["status"] == "completed" else "❌"
            lines.append(f"{icon} Step 3: Structured JSON Output")

            analysis = step.get("analysis_result", {})
            params = analysis.get("parameters", {})
            if params:
                lines.append("\n   HEALTH PARAMETERS:")
                for pname, pdata in params.items():
                    status = pdata.get("status", "Unknown")
                    value = pdata.get("value", 0)
                    unit = pdata.get("range", {}).get("unit", "")
                    s_icon = "✅" if status == "Normal" else "⚠️"
                    lines.append(f"   {s_icon} {pname.upper()}: {value} {unit} [{status}]")
                    if pdata.get("category"):
                        lines.append(f"      Category: {pdata['category']}")
            lines.append("")

        # Step 4 — Risk Prediction
        if "4_risk_prediction" in steps:
            step = steps["4_risk_prediction"]
            icon = "✅" if step["status"] == "completed" else "❌"
            lines.append(f"{icon} Step 4: Risk Prediction")

            if step["status"] == "completed":
                risk = step.get("risk_result", {})
                risk_level = risk.get("risk_level", "Unknown")
                risk_prob = risk.get("risk_percentage", "N/A")

                risk_emoji = {"Low": "✅", "Moderate": "⚠️",
                              "High": "⚠️⚠️"}.get(risk_level, "🚨")

                lines.append(f"   {risk_emoji} Risk Level: {risk_level}")
                lines.append(f"   Probability: {risk_prob}")
                lines.append(f"   Method: {risk.get('prediction_method', 'N/A')}")

                for factor in risk.get("risk_factors", []):
                    lines.append(
                        f"   • {factor['parameter'].upper()}: "
                        f"{factor['value']} [Impact: {factor['impact']}]"
                    )

                for rec in risk.get("recommendations", []):
                    lines.append(f"   • {rec}")
            else:
                lines.append(f"   Error: {step.get('error', 'Unknown error')}")
            lines.append("")

        # Step 5 — Symptom Checker
        if "5_symptom_checker" in steps:
            step = steps["5_symptom_checker"]
            icon = "✅" if step["status"] == "completed" else "❌"
            lines.append(f"{icon} Step 5: Symptom Checker")

            if step["status"] == "completed":
                lines.append(f"   Symptom Alignment: {step.get('symptom_alignment', 'N/A')}")
                lines.append(f"   Severity Score: {step.get('severity_score', 'N/A')}")
                lines.append(f"   Top Hypothesis: {step.get('top_hypothesis', 'N/A')}")
            else:
                lines.append(f"   Error: {step.get('error', 'Unknown error')}")
            lines.append("")

        # Step 5b — Trend Analysis
        if "5b_trend_analysis" in steps:
            step = steps["5b_trend_analysis"]
            icon = "✅" if step["status"] == "completed" else "❌"
            lines.append(f"{icon} Step 5b: Trend Analysis")
            if step["status"] == "completed":
                res = step.get("result", {})
                lines.append(f"   Direction: {res.get('direction', 'stable')}")
                lines.append(f"   Trend Score: {res.get('trend_score', 0.0)}")
                lines.append(f"   Visits Analyzed: {res.get('visits_used', 1)}")
            else:
                lines.append(f"   Warning/Error: {step.get('warning') or step.get('error', 'N/A')}")
            lines.append("")

        # Step 5c — Differential Analysis
        if "5c_differential_analysis" in steps:
            step = steps["5c_differential_analysis"]
            icon = "✅" if step["status"] == "completed" else "❌"
            lines.append(f"{icon} Step 5c: Differential Diagnosis")
            if step["status"] == "completed" and step.get("result"):
                diff = step["result"]
                lines.append(f"   Primary Hypothesis: {diff.get('primary', 'N/A')}")
                lines.append(f"   Shared Symptoms: {', '.join(diff.get('shared_symptoms', [])) or 'None'}")
            else:
                lines.append("   Differential Analysis: Single condition assessed or no alternatives.")
            lines.append("")

        # Step 6 — Result Fusion
        if "6_result_fusion" in steps:
            step = steps["6_result_fusion"]
            icon = "✅" if step["status"] == "completed" else "❌"
            lines.append(f"{icon} Step 6: Result Fusion")

            if step["status"] == "completed":
                merged = step.get("final_assessment", {})
                lines.append(f"   Winning Disease: {step.get('winning_disease', merged.get('disease', 'N/A'))}")
                lines.append(f"   Final Risk: {merged.get('risk_level', 'N/A')}")
                lines.append(f"   Final Score: {merged.get('score', 'N/A')}")
                lines.append(f"   Formula: {merged.get('formula', 'N/A')}")
                if "all_diseases" in step:
                    lines.append("   All Evaluated Diseases:")
                    for dname, dfusion in step["all_diseases"].items():
                        lines.append(f"     • {dname.title()}: Score={dfusion.get('score')} | Risk={dfusion.get('risk_level')}")
            else:
                lines.append(f"   Error: {step.get('error', 'Unknown error')}")
            lines.append("")

        # Step 6b — Explanation Generation
        if "6b_explanation_generation" in steps:
            step = steps["6b_explanation_generation"]
            icon = "✅" if step["status"] == "completed" else "❌"
            lines.append(f"{icon} Step 6b: Clinical Explanation")
            if step["status"] == "completed":
                expl = step.get("explanation", "")
                for eline in expl.split("\n"):
                    lines.append(f"   {eline}")
            else:
                lines.append(f"   Warning/Error: {step.get('warning') or step.get('error', 'N/A')}")
            lines.append("")

        # Step 7 — Alert Generation
        if "7_alert_generation" in steps:
            step = steps["7_alert_generation"]
            icon = "✅" if step["status"] == "completed" else "❌"
            lines.append(f"{icon} Step 7: Alert Generation")

            if step["status"] == "completed":
                alert_result = step.get("alert_result", {})
                alert = alert_result.get("alert", False)
                alert_icon = "⚠️" if alert else "✅"
                lines.append(f"   {alert_icon} Alert: {'Triggered' if alert else 'Not Triggered'}")
                lines.append(f"   Risk Level: {alert_result.get('risk_level', 'Unknown')}")
                lines.append(f"   Report: {alert_result.get('report', 'N/A')}")
                lines.append(f"   Notification: {alert_result.get('notification', 'N/A')}")
            else:
                lines.append(f"   Error: {step.get('error', 'Unknown error')}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    # ── unified entry point ───────────────────────────────────────────────

    def analyze(self, input_data: Any, input_type: str = "data") -> Dict[str, Any]:
        """Convenience wrapper: 'pdf' or 'data'."""
        if input_type == "pdf":
            return self.process_pdf_report(input_data)
        return self.process_health_data(input_data)


# ── standalone helper ─────────────────────────────────────────────────────────

def analyze_health(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone function to run complete health analysis.

    Usage:
        result = analyze_health({"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45})
    """
    agent = MasterHealthAgent()
    return agent.process_health_data(data)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "MASTER HEALTH AGENT TEST (LangGraph)" + " " * 17 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    test_data = {"hba1c": 6.8, "glucose": 148.0, "bmi": 29.0, "age": 45}
    print(f"Test Case: Prediabetic Patient\nInput Data: {test_data}\n")

    master = MasterHealthAgent()
    result = master.process_health_data(test_data)
    print(master.get_complete_report(result))

    # Additional scenarios
    print("\n\n" + "=" * 70)
    print("TESTING MULTIPLE SCENARIOS")
    print("=" * 70 + "\n")

    scenarios = [
        ("Healthy Individual", {"hba1c": 5.2, "glucose": 85, "bmi": 22, "age": 30}),
        ("High Risk Patient",  {"hba1c": 7.5, "glucose": 160, "bmi": 32, "age": 55}),
    ]

    for name, data in scenarios:
        print(f"\n{'─' * 70}")
        print(f"Scenario: {name}")
        print(f"{'─' * 70}")

        result = master.process_health_data(data)

        analysis = result.get("steps", {}).get("3_structured_output", {}).get("analysis_result", {})
        risk = result.get("steps", {}).get("4_risk_prediction", {})

        print(f"Status: {result['workflow_status']}")
        print(f"Abnormalities: {analysis.get('abnormal_count', 0)}")
        if risk.get("status") == "completed":
            print(f"Risk Level: {risk.get('risk_level')} ({risk.get('risk_probability')})")
        print()
