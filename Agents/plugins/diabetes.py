from typing import List, Dict, Any, Optional
import json
import logging
from .base import DiseasePlugin
from utils.helpers import normalize_level

logger = logging.getLogger(__name__)

class DiabetesPlugin(DiseasePlugin):
    name = "diabetes"
    required_fields = ["hba1c", "glucose", "bmi", "age"]
    optional_fields = ["homa_ir"]

    reference_ranges = {
        "hba1c": {"low": 4.0, "high": 5.6, "unit": "%"},
        "glucose": {"low": 70, "high": 99, "unit": "mg/dL"},
        "bmi": {"low": 18.5, "high": 24.9, "unit": "kg/m2"},
        "age": {"low": 0, "high": 120, "unit": "years"},
        "homa_ir": {"low": 0.0, "high": 2.5, "unit": "index"},
    }

    trend_scales = {
        "hba1c": 0.5,
        "glucose": 20.0,
        "bmi": 2.0
    }

    symptom_profiles = {
        "primary": ["polyuria", "polydipsia", "polyphagia", "unexplained weight loss", "fatigue"],
        "complications": {
            "cardiovascular": ["chest pain", "shortness of breath", "palpitations"],
            "cerebrovascular": ["slurred speech", "face droop", "one-side weakness"],
            "renal": ["foamy urine", "reduced urine output", "swelling in ankles"],
            "retinopathy": ["blurred vision", "dark spots", "vision loss"]
        }
    }

    alert_thresholds = {
        "hba1c": 7.0,
        "glucose": 200.0
    }

    def flags(self, data: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
        flags = []
        hba1c = data.get("hba1c")
        if hba1c is not None:
            try:
                if float(hba1c) >= 6.5:
                    flags.append("DIABETES CRITERIA MET")
                elif float(hba1c) >= 5.7:
                    flags.append("PRE-DIABETES RANGE")
            except (ValueError, TypeError):
                pass

        homa_ir = data.get("homa_ir")
        if homa_ir is not None:
            try:
                if float(homa_ir) > 2.5:
                    flags.append("INSULIN RESISTANCE")
            except (ValueError, TypeError):
                pass
        elif params and params.get("homa_ir", {}).get("status") == "High":
            flags.append("INSULIN RESISTANCE")

        return flags

    def notes(self, param: str, status: str) -> str:
        notes_map = {
            "hba1c": {"High": "Diabetic range (ADA 6.5%+)", "Low": "Hypoglycemia risk"},
            "glucose": {"High": "FPG diabetes criterion (≥126 mg/dL)", "Low": "Low blood sugar"},
            "bmi": {"High": "Overweight/Obese range", "Low": "Underweight range"},
            "homa_ir": {"High": "Insulin resistance marker (>2.5)", "Low": "Normal"},
        }
        return notes_map.get(param, {}).get(status, "Check clinical guidelines")

    def rule_risk(self, data: Dict[str, Any], analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from Agents.riskpredictor import compute_diabetes_rule_risk
        res = compute_diabetes_rule_risk(data, analysis)
        raw_level = res.get("risk_level") or "Low"
        norm_level = normalize_level(raw_level)
        score = res.get("risk_score")
        if score is None:
            score = 0.0
        prob = res.get("risk_probability")
        if prob is None:
            prob = round(score / 100.0, 2)

        return {
            "score": score,
            "level": norm_level,
            "risk_level": norm_level,
            "risk_probability": prob,
            "risk_percentage": f"{int(round(score))}%",
            "method": "rule-based",
            "summary": res.get("summary", "")
        }

    def predict_risk(self, data: Dict[str, Any], analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            from Agents.riskpredictor import RiskPredictorAgent
            predictor = RiskPredictorAgent()
            risk_json = predictor.predict_from_analysis(analysis or data)
            risk = json.loads(risk_json)
            if risk.get("success") is not False and not risk.get("error"):
                risk["risk_level"] = normalize_level(risk.get("risk_level", "Low"))
                return risk
        except Exception as e:
            logger.debug(f"RiskPredictorAgent failed in DiabetesPlugin: {e}")

        return self.rule_risk(data, analysis)

    def score_symptoms(self, symptoms: List[str]) -> Dict[str, Any]:
        match_count = 0
        primary_symptoms = self.symptom_profiles["primary"]
        for s in symptoms:
            if any(ps in s.lower() for ps in primary_symptoms):
                match_count += 1

        severity = min(match_count / max(len(primary_symptoms), 1), 1.0)
        return {
            "severity": severity,
            "severity_score": severity,
            "top_hypothesis": "Diabetes Mellitus" if severity > 0.3 else "Unknown"
        }
