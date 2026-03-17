"""
Report Analyzer Agent — LangChain Implementation

Purpose:
  Checks lab values and identifies abnormal parameters
  before sending data to the ML model.

Steps:
  1. Define reference ranges
  2. Validate parameters against ranges
  3. Expose as LangChain @tool
  4. Create LangChain Agent (initialize_agent + Ollama)
  5. Run the agent
"""

import logging
import json
import ast
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

from langchain.tools import tool
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from utils.constants import REFERENCE_RANGES

logger = logging.getLogger(__name__)


UNITS = {
    "hba1c": "%",
    "glucose": "mg/dL",
    "bmi": "kg/m²",
    "age": "years",
    "homa_ir": "index",
}

NOTES = {
    ("hba1c", "High"): "Diabetic range (ADA 6.5%+)",
    ("glucose", "High"): "FPG diabetes criterion (≥126 mg/dL)",
    ("homa_ir", "High"): "Insulin resistance marker (>2.5)",
}

SUPPLEMENTAL_RANGES = {
    # HOMA-IR is optional in upstream extraction; include a practical threshold.
    "homa_ir": {"low": 0.0, "high": 2.5, "unit": "index"},
}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Reference Ranges  (imported from utils.constants)
#
#  REFERENCE_RANGES = {
#      "hba1c":   {"low": 4.0,  "high": 5.6,  ...},
#      "glucose": {"low": 70,   "high": 99,   ...},
#      "bmi":     {"low": 18.5, "high": 24.9, ...},
#      "age":     {"low": 0,    "high": 120,  ...},
#  }
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Validation Function
# ══════════════════════════════════════════════════════════════════════════════

def validate_parameters(data: dict) -> dict:
    """
    Validate health parameters against reference ranges.

    Args:
        data: e.g. {"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45}

    Returns:
        {
          "parameters": {
                        "hba1c":   {"value": 6.8, "status": "High", "unit": "%", "note": "Diabetic range (ADA 6.5%+)"},
                        "glucose": {"value": 148, "status": "High", "unit": "mg/dL", "note": "FPG diabetes criterion (≥126 mg/dL)"},
            ...
          },
          "abnormal_count": 3,
                    "abnormal_parameters": ["hba1c", "glucose", "bmi"],
                    "endocrine_flags": ["DIABETES CRITERIA MET", "INSULIN RESISTANCE"]
        }
    """
    results = {}
    abnormal = 0
    abnormal_params = []

    for k, v in data.items():
        r = REFERENCE_RANGES.get(k) or SUPPLEMENTAL_RANGES.get(k)

        if not r:
            results[k] = {
                "value": v,
                "status": "Unknown",
                "unit": "",
                "note": "No reference range available",
            }
            continue

        if v < r["low"]:
            status = "Low"
            abnormal += 1
            abnormal_params.append(k)
        elif v > r["high"]:
            status = "High"
            abnormal += 1
            abnormal_params.append(k)
        else:
            status = "Normal"

        results[k] = {
            "value": v,
            "status": status,
            "unit": UNITS.get(k, r.get("unit", "")),
            "note": NOTES.get((k, status), ""),
        }

    flags = []
    hba1c_val = data.get("hba1c")
    if isinstance(hba1c_val, (int, float)) and hba1c_val >= 6.5:
        flags.append("DIABETES CRITERIA MET")

    homa_ir_val = data.get("homa_ir")
    if isinstance(homa_ir_val, (int, float)) and homa_ir_val > 2.5:
        flags.append("INSULIN RESISTANCE")

    return {
        "parameters": results,
        "abnormal_count": abnormal,
        "abnormal_parameters": abnormal_params,
        "endocrine_flags": flags,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — LangChain Tool
# ══════════════════════════════════════════════════════════════════════════════

@tool
def analyze_lab_values(data: dict) -> dict:
    """
    Analyze health lab values and detect abnormalities.

    Accepts a dict with keys: hba1c, glucose, bmi, age.
    Returns parameter statuses (Normal / Low / High) and the abnormal count.

    Args:
        data: Health parameters, e.g. {"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45}

    Returns:
        Analysis result with parameter statuses and abnormal count.
    """
    return validate_parameters(data)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — LangChain Agent
# ══════════════════════════════════════════════════════════════════════════════

def create_analyzer_agent():
    """
    Build and return a LangGraph ReAct agent wired to the analyze_lab_values tool.
    """
    llm = ChatOllama(model="mistral", temperature=0)
    agent = create_react_agent(llm, tools=[analyze_lab_values])
    return agent


def _extract_analyzer_tool_payload(agent_response: Dict[str, Any]) -> Dict[str, Any]:
    """Extract analyze_lab_values tool output from a LangGraph response."""
    messages = agent_response.get("messages", []) if isinstance(agent_response, dict) else []
    for msg in reversed(messages):
        if getattr(msg, "name", "") != "analyze_lab_values":
            continue

        content = getattr(msg, "content", "")
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                try:
                    parsed = ast.literal_eval(content)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    continue
    return {}


def analyze_with_langgraph_agent(data: Dict[str, float]) -> Dict[str, Any]:
    """Run analyzer through the LangGraph ReAct agent and return structured output."""
    agent = create_analyzer_agent()
    prompt = (
        "Use the analyze_lab_values tool exactly once with this payload and return only that result: "
        f"{json.dumps(data)}"
    )
    response = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    return _extract_analyzer_tool_payload(response)


# ══════════════════════════════════════════════════════════════════════════════
#  Backward-compatible class wrapper
#  (used by masterhealth.py and healthroutes.py)
# ══════════════════════════════════════════════════════════════════════════════

class ReportAnalyzerAgent:
    """
    Thin class wrapper around validate_parameters().
    Keeps the same interface consumed by MasterHealthAgent and API routes.
    """

    def __init__(self):
        self.reference_ranges = REFERENCE_RANGES
        logger.info("Report Analyzer Agent initialized")

    def analyze_health_record(self, health_data: Dict[str, float], use_agent: bool = False) -> str:
        """
        Main analysis method — returns a JSON string.

        Args:
            health_data: {"hba1c": 7.8, "glucose": 148, "bmi": 29, "age": 45}

        Returns:
            JSON string of analysis result
        """
        try:
            logger.info("Starting health record analysis")
            if use_agent:
                try:
                    result = analyze_with_langgraph_agent(health_data)
                    if not result:
                        logger.warning("Agent returned no structured payload; falling back to direct validation")
                        result = validate_parameters(health_data)
                except Exception as agent_error:
                    logger.warning(f"Agent analysis failed, falling back to direct validation: {agent_error}")
                    result = validate_parameters(health_data)
            else:
                result = validate_parameters(health_data)
            result["analysis_timestamp"] = str(datetime.now())
            result["analyzer_version"] = "2.0"
            logger.info("Analysis completed successfully")
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return json.dumps({"error": str(e), "status": "failed"})

    def analyze_pdf_report(self, pdf_path: str, use_llm: bool = True) -> str:
        """Analyze health data extracted from a PDF lab report."""
        try:
            from tools.labparse import extract_health_data_from_pdf

            logger.info(f"Analyzing PDF report: {Path(pdf_path).name}")
            health_data = extract_health_data_from_pdf(pdf_path, use_llm=use_llm)
            filtered = {k: v for k, v in health_data.items() if v is not None}

            if not filtered:
                return json.dumps({
                    "error": "No health parameters could be extracted from PDF",
                    "status": "failed",
                    "file": Path(pdf_path).name,
                })
            return self.analyze_health_record(filtered)
        except FileNotFoundError:
            return json.dumps({"error": f"PDF file not found: {pdf_path}", "status": "failed"})
        except Exception as e:
            logger.error(f"PDF analysis failed: {str(e)}")
            return json.dumps({"error": str(e), "status": "failed", "file": Path(pdf_path).name})

    def get_summary(self, analysis_result: Dict[str, Any]) -> str:
        """Generate a human-readable summary."""
        abnormal_count = analysis_result.get("abnormal_count", 0)
        if abnormal_count == 0:
            return "All parameters are within normal range."
        abnormal_params = ", ".join(analysis_result.get("abnormal_parameters", []))
        return f"⚠️ {abnormal_count} abnormal parameter(s) detected: {abnormal_params}"


# Standalone helper
def analyze_health_data(data: Dict[str, float]) -> Dict[str, Any]:
    """Quick standalone analysis — returns a dict."""
    agent = ReportAnalyzerAgent()
    return json.loads(agent.analyze_health_record(data))


__all__ = [
    "ReportAnalyzerAgent",
    "validate_parameters",
    "analyze_lab_values",
    "analyze_health_data",
    "analyze_with_langgraph_agent",
    "create_analyzer_agent",
]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Run the agent
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Sample input
    data = {
        "hba1c": 5.8,
        "glucose": 168.0,
        "bmi": 23.0,
        "age": 55.0,
    }

    # --- Direct validation (no LLM needed) ---
    print("Direct validation")
    print("=" * 50)
    result = validate_parameters(data)
    print(json.dumps(result, indent=2))

    # --- Via LangChain Agent (requires Ollama running locally) ---
    try:
        print("\nLangChain Agent")
        print("=" * 50)
        agent_result = analyze_with_langgraph_agent(data)
        if not agent_result:
            agent_result = validate_parameters(data)
        print(json.dumps(agent_result, indent=2))
    except Exception as e:
        print(f"LangChain agent skipped ({e})")
