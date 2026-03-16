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
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

from langchain.tools import tool
from langchain_community.llms import Ollama
from langgraph.prebuilt import create_react_agent

from utils.constants import REFERENCE_RANGES

logger = logging.getLogger(__name__)


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
            "hba1c":   {"value": 6.8, "status": "High"},
            "glucose": {"value": 148, "status": "High"},
            ...
          },
          "abnormal_count": 3,
          "abnormal_parameters": ["hba1c", "glucose", "bmi"]
        }
    """
    results = {}
    abnormal = 0
    abnormal_params = []

    for k, v in data.items():
        r = REFERENCE_RANGES[k]

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

        results[k] = {"value": v, "status": status}

    return {
        "parameters": results,
        "abnormal_count": abnormal,
        "abnormal_parameters": abnormal_params,
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
    llm = Ollama(model="mistral")
    agent = create_react_agent(llm, tools=[analyze_lab_values])
    return agent


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

    def analyze_health_record(self, health_data: Dict[str, float]) -> str:
        """
        Main analysis method — returns a JSON string.

        Args:
            health_data: {"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45}

        Returns:
            JSON string of analysis result
        """
        try:
            logger.info("Starting health record analysis")
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
        agent = create_analyzer_agent()
        agent_result = agent.run(
            f"Analyze these lab values and detect abnormalities: {json.dumps(data)}"
        )
        print(agent_result)
    except Exception as e:
        print(f"LangChain agent skipped ({e})")
