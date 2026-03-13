"""Simple end-to-end pipeline script for the AI Health agents.

This script simulates a real usage flow for a single patient case,
executing the agents in this order:

1. Report Analyzer Agent  – validate lab parameters and identify abnormal values
2. Risk Prediction Agent  – estimate diabetes probability (rule-based fallback)
3. Symptom Checker Agent  – analyze symptoms and compute alignment/severity
4. Master-style Fusion    – combine lab, risk, and symptom signals
5. Alert Agent            – generate final alert + patient-friendly report

Run from the project root:

    python test-agents-pipeline.py

The script avoids LLM calls by using rule-based fallbacks where possible,
so it should run without Ollama or external API keys.
"""

from __future__ import annotations

import json
from pprint import pprint

from Agents.reportanalyzer import analyze_health_data
from Agents.masterhealth import (
    _rule_based_risk,
    _to_probability,
    _derive_symptom_metrics,
    _score_to_risk_level,
)
from Agents.symptomchecker import analyze_symptoms
from Agents.alertsystem import AlertAgent


def run_pipeline() -> None:
    # ------------------------------------------------------------------
    # Sample patient input (same profile used in README)
    # ------------------------------------------------------------------
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
    manual_text = "Family history of diabetes; symptoms worsening over 3 months."

    print("\n================= TEST AGENTS PIPELINE =================\n")
    print("Input health data:")
    pprint(health_data)
    print("\nInput symptoms:")
    pprint(symptoms)

    # ------------------------------------------------------------------
    # 1️⃣ Report Analyzer Agent
    # ------------------------------------------------------------------
    print("\n[1] REPORT ANALYZER AGENT")
    analysis = analyze_health_data(health_data)

    abnormal_count = analysis.get("abnormal_count", 0)
    abnormal_params = analysis.get("abnormal_parameters", [])

    print("Abnormal parameter count:", abnormal_count)
    print("Abnormal parameters:", abnormal_params)
    print("Parameter statuses:")
    for name, info in analysis.get("parameters", {}).items():
        print(f"  - {name}: {info.get('value')} [{info.get('status')}]")

    # ------------------------------------------------------------------
    # 2️⃣ Risk Prediction Agent (rule-based fallback from MasterHealthAgent)
    # ------------------------------------------------------------------
    print("\n[2] RISK PREDICTION AGENT (rule-based)")
    risk_result = _rule_based_risk(analysis, health_data)

    print("Risk level:", risk_result.get("risk_level"))
    print("Risk probability (string):", risk_result.get("risk_probability"))

    ml_probability = _to_probability(
        risk_result.get("risk_probability") or risk_result.get("risk_percentage")
    ) or 0.0
    print("Risk probability (0-1):", ml_probability)

    # ------------------------------------------------------------------
    # 3️⃣ Symptom Checker Agent
    # ------------------------------------------------------------------
    print("\n[3] SYMPTOM CHECKER AGENT")
    symptom_result = analyze_symptoms(
        symptoms=symptoms,
        report_context=analysis,
        risk_context=risk_result,
        manual_text=manual_text,
        use_llm=False,  # force rule-based reasoning for reproducibility
    )

    # Derive alignment + normalized severity score exactly as MasterHealthAgent does
    symptom_result.update(_derive_symptom_metrics(symptom_result))

    print("Symptom alignment:", symptom_result.get("symptom_alignment"))
    print("Symptom severity score (0-1):", symptom_result.get("severity_score"))
    print("Top hypothesis:", symptom_result.get("input_summary", {}).get("top_hypothesis"))

    # ------------------------------------------------------------------
    # 4️⃣ Master-style result fusion (same formula as MasterHealthAgent)
    # ------------------------------------------------------------------
    print("\n[4] RESULT FUSION (MASTER HEALTH LOGIC)")

    abnormal_weight = min(max(int(abnormal_count), 0) / 3.0, 1.0)
    symptom_score = float(symptom_result.get("severity_score") or 0.0)
    symptom_score = max(0.0, min(1.0, symptom_score))

    final_score = round(
        (ml_probability * 0.6)
        + (abnormal_weight * 0.3)
        + (symptom_score * 0.1),
        4,
    )
    final_risk_level = _score_to_risk_level(final_score)

    final_assessment = {
        "risk_level": final_risk_level,
        "score": final_score,
        "ml_probability": ml_probability,
        "abnormal_count": abnormal_count,
        "abnormal_count_weight": round(abnormal_weight, 4),
        "symptom_score": round(symptom_score, 4),
    }

    print("Final fused risk level:", final_assessment["risk_level"])
    print("Final fused score:", final_assessment["score"])
    print("Components:")
    print("  - ML probability:", final_assessment["ml_probability"])
    print("  - Abnormal count weight:", final_assessment["abnormal_count_weight"])
    print("  - Symptom score:", final_assessment["symptom_score"])

    # ------------------------------------------------------------------
    # 5️⃣ Alert Agent – final alert + patient report
    # ------------------------------------------------------------------
    print("\n[5] ALERT AGENT")

    alert_input = {
        "health_data": health_data,
        "analysis_result": analysis,
        "risk_result": risk_result,
        "diabetes_probability": final_assessment["score"],
        "abnormal_parameters": abnormal_count,
        "symptom_score": final_assessment["symptom_score"],
    }

    alert_agent = AlertAgent(use_llm=False)  # fallback text-only report
    alert_result = alert_agent.process(alert_input)

    print("Alert triggered:", alert_result["alert"])
    print("Alert risk level:", alert_result["risk_level"])
    print("Alert triggers:")
    for trig in alert_result.get("triggers", []):
        print("  -", trig)

    print("\nPatient-facing report:\n")
    print(alert_result["report"])

    print("\nRaw alert payload (for debugging):")
    print(json.dumps(alert_result, indent=2))


if __name__ == "__main__":
    run_pipeline()
