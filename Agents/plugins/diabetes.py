from typing import List, Dict, Any, Optional
from Agents.plugins.base import DiseasePlugin
from Agents.plugins.registry import register
from utils.constants import REFERENCE_RANGES

class DiabetesPlugin(DiseasePlugin):
    name = "diabetes"
    required_fields = ["hba1c", "glucose", "bmi", "age"]
    optional_fields = ["homa_ir"]

    reference_ranges = {
        **REFERENCE_RANGES,
        "homa_ir": {"low": 0.0, "high": 2.5, "unit": "index"}
    }

    param_alert_thresholds = {"hba1c": 7.0}

    # In the plan, these are defined in the wrapper to maintain existing logic
    # but we use the project constants for ranges/notes.
    from Agents.reportanalyzer import UNITS, NOTES

    def flags(self, data: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
        flags = []
        hba1c_val = data.get("hba1c")
        if isinstance(hba1c_val, (int, float)) and hba1c_val >= 6.5:
            flags.append("DIABETES CRITERIA MET")

        homa_ir_val = data.get("homa_ir")
        if isinstance(homa_ir_val, (int, float)) and homa_ir_val > 2.5:
            flags.append("INSULIN RESISTANCE")
        return flags

    def note(self, param: str, status: str) -> str:
        from Agents.reportanalyzer import NOTES
        return NOTES.get((param, status), "")

    def rule_risk(self, data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        # To be implemented by Shreya (per Step C)
        return {"risk_level": "Unknown", "risk_probability": 0.0}

    def predict_risk(self, data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        # To be implemented by Shreya (per Step C)
        return {"risk_level": "Unknown", "risk_probability": 0.0}

    def score_symptoms(self, symptoms: List[str]) -> Dict[str, Any]:
        # To be implemented by Vaibhavi (per Step C)
        return {"severity_score": 0.0, "top_hypothesis": "None"}

# Register the plugin
register(DiabetesPlugin())
