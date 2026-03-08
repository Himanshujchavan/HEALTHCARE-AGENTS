"""
Master Health Agent — LangGraph Orchestrator
Routes health analysis tasks to sub-agents via a compiled LangGraph StateGraph.

Workflow:
  START ─→ [extract_pdf] ─→ analyze ─→ [predict_risk] ─→ finalize ─→ END

Nodes:
  extract_pdf   – Extracts health data from a PDF lab report (skipped for dict input)
  analyze       – Validates parameters & detects abnormalities (ReportAnalyzerAgent)
  predict_risk  – Predicts diabetes/metabolic risk (LLM → rule-based fallback)
  finalize      – Marks the workflow as completed
"""

import logging
import json
import re
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict

# Import sub-agents
from Agents.reportanalyzer import ReportAnalyzerAgent

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  WORKFLOW STATE
# ══════════════════════════════════════════════════════════════════════════════

class HealthWorkflowState(TypedDict):
    """Typed state that flows through the LangGraph workflow."""
    health_data: Dict[str, float]
    include_risk: bool
    input_type: str                 # "data" | "pdf"
    pdf_path: Optional[str]
    use_llm: bool
    analysis_result: Optional[Dict]
    risk_result: Optional[Dict]
    workflow_status: str
    error: Optional[str]
    steps: Dict[str, Any]
    timestamp: str
    input_file: Optional[str]


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
    """Validate parameters & detect abnormalities via ReportAnalyzerAgent."""
    try:
        logger.info("NODE analyze: ReportAnalyzerAgent")
        analyzer = ReportAnalyzerAgent()
        analysis_json = analyzer.analyze_health_record(state["health_data"])
        analysis = json.loads(analysis_json)

        if "error" in analysis:
            return {
                "workflow_status": "failed",
                "error": analysis["error"],
                "steps": {**state["steps"],
                          "1_parameter_validation": {"status": "failed",
                                                     "error": analysis["error"]}},
            }

        updated_steps = {**state["steps"]}
        updated_steps["1_parameter_validation"] = {
            "status": "completed",
            "validated_parameters": list(analysis.get("parameters", {}).keys()),
        }
        updated_steps["2_abnormality_detection"] = {
            "status": "completed",
            "abnormal_count": analysis.get("abnormal_count", 0),
            "abnormal_parameters": analysis.get("abnormal_parameters", []),
        }
        updated_steps["3_structured_output"] = {
            "status": "completed",
            "format": "JSON",
            "analysis_result": analysis,
        }

        return {"analysis_result": analysis, "steps": updated_steps}
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {"workflow_status": "failed", "error": str(e)}


def predict_risk_node(state: HealthWorkflowState) -> dict:
    """Predict health risk — tries dedicated sub-agent, then LLM, then rules."""
    try:
        logger.info("NODE predict_risk")
        analysis = state.get("analysis_result", {})
        health_data = state.get("health_data", {})

        # 1. Try the dedicated RiskPredictorAgent (may be empty / not implemented)
        try:
            from Agents.riskpredictor import RiskPredictorAgent
            predictor = RiskPredictorAgent()
            risk_json = predictor.predict_from_analysis(analysis)
            risk = json.loads(risk_json)
        except (ImportError, AttributeError, TypeError):
            # 2. Fall back to LangChain LLM risk prediction
            risk = _llm_risk_prediction(analysis, health_data)

        if "error" in risk:
            return {
                "risk_result": risk,
                "steps": {**state["steps"], "4_risk_prediction": {
                    "status": "failed", "error": risk["error"]
                }},
            }

        return {
            "risk_result": risk,
            "steps": {**state["steps"], "4_risk_prediction": {
                "status": "completed",
                "risk_level": risk.get("risk_level"),
                "risk_probability": risk.get("risk_probability"),
                "risk_result": risk,
            }},
        }
    except Exception as e:
        logger.error(f"Risk prediction failed: {e}")
        return {
            "risk_result": {"error": str(e)},
            "steps": {**state["steps"], "4_risk_prediction": {
                "status": "failed", "error": str(e)
            }},
        }


def finalize_node(state: HealthWorkflowState) -> dict:
    """Mark the workflow as completed (unless already failed)."""
    if state.get("workflow_status") != "failed":
        return {"workflow_status": "completed"}
    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  LLM + RULE-BASED RISK PREDICTION (fallback when sub-agent is missing)
# ══════════════════════════════════════════════════════════════════════════════

def _llm_risk_prediction(analysis: Dict, health_data: Dict) -> Dict[str, Any]:
    """Use ChatOpenAI for risk prediction; falls back to rules on failure."""
    try:
        llm = Ollama(model="mistral")

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a medical risk assessment AI.\n"
             "Given a patient's health analysis, predict their diabetes/metabolic risk.\n\n"
             "Return ONLY valid JSON with this structure:\n"
             '{{\n'
             '  "risk_level": "Low" | "Moderate" | "High" | "Critical",\n'
             '  "risk_probability": "XX%",\n'
             '  "risk_percentage": "XX%",\n'
             '  "prediction_method": "llm_analysis",\n'
             '  "risk_factors": [\n'
             '    {{"parameter": "name", "value": 0, "impact": "Low|Medium|High"}}\n'
             '  ],\n'
             '  "recommendations": ["recommendation1", "recommendation2"]\n'
             '}}'),
            ("user",
             "Patient Health Data: {health_data}\n\n"
             "Analysis Result: {analysis}\n\n"
             "Provide risk assessment as JSON only."),
        ])

        chain = prompt | llm
        response = chain.invoke({
            "health_data": json.dumps(health_data),
            "analysis": json.dumps(analysis, default=str),
        })

        content = response.strip() if isinstance(response, str) else response.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = re.sub(r"```json?\s*", "", content)
            content = re.sub(r"```\s*$", "", content)

        return json.loads(content)
    except Exception as e:
        logger.warning(f"LLM risk prediction unavailable ({e}), using rule-based fallback")
        return _rule_based_risk(analysis, health_data)


def _rule_based_risk(analysis: Dict, health_data: Dict) -> Dict[str, Any]:
    """Deterministic rule-based risk scoring (always available)."""
    risk_factors = []
    recommendations = []

    hba1c = health_data.get("hba1c", 0)
    glucose = health_data.get("glucose", 0)
    bmi = health_data.get("bmi", 0)
    age = health_data.get("age", 0)

    score = 0

    if hba1c >= 6.5:
        score += 3
        risk_factors.append({"parameter": "hba1c", "value": hba1c, "impact": "High"})
        recommendations.append("Consult endocrinologist for diabetes management")
    elif hba1c >= 5.7:
        score += 2
        risk_factors.append({"parameter": "hba1c", "value": hba1c, "impact": "Medium"})
        recommendations.append("Monitor HbA1c every 3 months")

    if glucose >= 126:
        score += 3
        risk_factors.append({"parameter": "glucose", "value": glucose, "impact": "High"})
        recommendations.append("Regular fasting glucose monitoring recommended")
    elif glucose >= 100:
        score += 2
        risk_factors.append({"parameter": "glucose", "value": glucose, "impact": "Medium"})

    if bmi >= 30:
        score += 2
        risk_factors.append({"parameter": "bmi", "value": bmi, "impact": "High"})
        recommendations.append("Weight management program recommended")
    elif bmi >= 25:
        score += 1
        risk_factors.append({"parameter": "bmi", "value": bmi, "impact": "Medium"})

    if age >= 45:
        score += 1
        risk_factors.append({"parameter": "age", "value": age, "impact": "Medium"})

    if not recommendations:
        recommendations.append("Maintain healthy lifestyle and regular checkups")

    if score <= 1:
        risk_level, prob = "Low", "15%"
    elif score <= 3:
        risk_level, prob = "Moderate", "40%"
    elif score <= 5:
        risk_level, prob = "High", "65%"
    else:
        risk_level, prob = "Critical", "85%"

    return {
        "risk_level": risk_level,
        "risk_probability": prob,
        "risk_percentage": prob,
        "prediction_method": "rule_based",
        "risk_factors": risk_factors,
        "recommendations": recommendations,
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
    """After analysis: predict risk or go straight to finalize."""
    if state.get("workflow_status") == "failed":
        return "finalize"
    if state.get("include_risk", True):
        return "predict_risk"
    return "finalize"


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
        "finalize": "finalize",
        "predict_risk": "predict_risk",
    })
    graph.add_edge("predict_risk", "finalize")
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

    def process_health_data(self, health_data: Dict[str, float],
                            include_risk: bool = True) -> Dict[str, Any]:
        """
        Process health data through the LangGraph workflow.

        Args:
            health_data: {"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45}
            include_risk: Whether to include risk prediction step
        """
        initial_state: HealthWorkflowState = {
            "health_data": health_data,
            "include_risk": include_risk,
            "input_type": "data",
            "pdf_path": None,
            "use_llm": False,
            "analysis_result": None,
            "risk_result": None,
            "workflow_status": "in_progress",
            "error": None,
            "steps": {},
            "timestamp": str(datetime.now()),
            "input_file": None,
        }

        result = self.workflow.invoke(initial_state)

        return {
            "workflow_status": result["workflow_status"],
            "timestamp": result["timestamp"],
            "steps": result["steps"],
            **({"error": result["error"]} if result.get("error") else {}),
        }

    # ── process PDF input ─────────────────────────────────────────────────

    def process_pdf_report(self, pdf_path: str,
                           use_llm: bool = False,
                           include_risk: bool = True) -> Dict[str, Any]:
        """
        Process a PDF lab report through the LangGraph workflow.

        Args:
            pdf_path: Path to the PDF file
            use_llm: Use LLM for PDF text extraction
            include_risk: Whether to include risk prediction step
        """
        initial_state: HealthWorkflowState = {
            "health_data": {},
            "include_risk": include_risk,
            "input_type": "pdf",
            "pdf_path": pdf_path,
            "use_llm": use_llm,
            "analysis_result": None,
            "risk_result": None,
            "workflow_status": "in_progress",
            "error": None,
            "steps": {},
            "timestamp": str(datetime.now()),
            "input_file": Path(pdf_path).name,
        }

        result = self.workflow.invoke(initial_state)

        return {
            "workflow_status": result["workflow_status"],
            "timestamp": result["timestamp"],
            "input_type": "pdf",
            "input_file": result.get("input_file", Path(pdf_path).name),
            "steps": result["steps"],
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

        lines.append("=" * 70)
        return "\n".join(lines)

    # ── unified entry point ───────────────────────────────────────────────

    def analyze(self, input_data: Any, input_type: str = "data") -> Dict[str, Any]:
        """Convenience wrapper: 'pdf' or 'data'."""
        if input_type == "pdf":
            return self.process_pdf_report(input_data)
        return self.process_health_data(input_data)


# ── standalone helper ─────────────────────────────────────────────────────────

def analyze_health(data: Dict[str, float]) -> Dict[str, Any]:
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
