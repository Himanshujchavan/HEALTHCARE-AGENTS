"""
Report Analyzer Agent — Rule-based Implementation

Purpose:
    Checks lab values and identifies abnormal parameters
    before sending data to the ML model.

Steps:
    1. Define reference ranges
    2. Validate parameters against ranges
    3. Expose as LangChain @tool
    4. Return structured output
"""

import logging
import json
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

from langchain.tools import tool

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

def validate_parameters(data: dict, plugin=None) -> dict:
    """
    Validate health parameters against reference ranges.

    Args:
        data: e.g. {"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45}
        plugin: Optional DiseasePlugin for custom ranges and flags.
    """
    if plugin is None:
        from Agents.plugins.diabetes import DiabetesPlugin
        plugin = DiabetesPlugin()

    results = {}
    abnormal = 0
    abnormal_params = []

    valid_keys = set(plugin.required_fields + getattr(plugin, "optional_fields", []))

    for k, v in data.items():
        if k not in valid_keys:
            continue
        if v is None:
            continue

        if isinstance(v, bool):
            # Categorical parameters (e.g. smoker) are reported, not ranged.
            results[k] = {
                "value": v,
                "status": "Reported",
                "unit": "",
                "note": plugin.notes(k, "Reported"),
            }
            continue

        try:
            val_num = float(v)
        except (ValueError, TypeError):
            continue

        r = plugin.reference_ranges.get(k)

        if not r:
            results[k] = {
                "value": val_num,
                "status": "Unknown",
                "unit": "",
                "note": "No reference range available",
            }
            continue

        if r.get("categorical"):
            # An ordinal category code is not a measurement: reporting it as
            # Normal/High against a numeric range would be meaningless.
            results[k] = {
                "value": val_num,
                "status": "Reported",
                "unit": r.get("unit", ""),
                "note": plugin.notes(k, "Reported"),
            }
            continue

        if val_num < r["low"]:
            status = "Low"
            abnormal += 1
            abnormal_params.append(k)
        elif val_num > r["high"]:
            status = "High"
            abnormal += 1
            abnormal_params.append(k)
        else:
            status = "Normal"

        results[k] = {
            "value": val_num,
            "status": status,
            "unit": UNITS.get(k, r.get("unit", "")),
            "note": plugin.notes(k, status),
        }

    flags = plugin.flags(data, results)

    return {
        "parameters": results,
        "abnormal_count": abnormal,
        "abnormal_parameters": abnormal_params,
        # "flags" is the disease-neutral key; "endocrine_flags" is kept as an
        # alias for existing diabetes consumers.
        "flags": flags,
        "endocrine_flags": flags,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — LangChain Tool
# ══════════════════════════════════════════════════════════════════════════════

@tool
def analyze_lab_values(data: dict, plugin=None) -> dict:
    """
    Analyze health lab values and detect abnormalities.

    Accepts a dict with keys: hba1c, glucose, bmi, age.
    Returns parameter statuses (Normal / Low / High) and the abnormal count.

    Args:
        data: Health parameters, e.g. {"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45}
        plugin: Optional DiseasePlugin for custom ranges and flags.
    """
    return validate_parameters(data, plugin=plugin)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — LLM-Disabled Compatibility Shims
# ══════════════════════════════════════════════════════════════════════════════

def create_analyzer_agent():
    """LLM analyzer disabled; retained for backward compatibility."""
    logger.info("LLM analyzer disabled; using rule-based validation.")
    return None


def analyze_with_langgraph_agent(data: Dict[str, float], plugin=None) -> Dict[str, Any]:
    """LLM analyzer disabled; return rule-based validation output."""
    logger.info("LLM analyzer disabled; using rule-based validation.")
    return validate_parameters(data, plugin=plugin)


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

    def analyze_health_record(self, health_data: Dict[str, float], use_agent: bool = False, plugin=None) -> str:
        """
        Main analysis method — returns a JSON string.

        Args:
            health_data: {"hba1c": 7.8, "glucose": 148, "bmi": 29, "age": 45}
            use_agent: Whether to use LLM analyzer (currently disabled)
            plugin: Optional DiseasePlugin for custom ranges and flags.
        """
        try:
            logger.info("Starting health record analysis")
            if use_agent:
                logger.info("use_agent requested, but LLM analyzer is disabled; using rule-based validation")
            result = validate_parameters(health_data, plugin=plugin)
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
#  STEP 5 — Run the analyzer
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

    # --- Agent mode (LLM disabled, rule-based) ---
    print("\nAgent mode (LLM disabled)")
    print("=" * 50)
    agent_result = analyze_with_langgraph_agent(data)
    print(json.dumps(agent_result, indent=2))
