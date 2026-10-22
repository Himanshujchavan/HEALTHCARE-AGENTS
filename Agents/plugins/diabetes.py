"""Diabetes plugin.

Wraps the existing diabetes logic (report analyzer ranges, rule scorer and
symptom checker) behind the common ``DiseasePlugin`` interface and emits the
canonical risk contract shared by every plugin.
"""

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

        risk_factors, recommendations = self.risk_factors(data)

        return {
            "status": "completed",
            "disease": self.name,
            "score": score,
            "level": norm_level,
            "risk_level": norm_level,
            "risk_probability": prob,
            "risk_score": score,
            "risk_percentage": f"{int(round(score))}%",
            "method": "rule_based",
            "prediction_method": "rule_based",
            "risk_factors": risk_factors,
            "recommendations": recommendations,
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
                risk["status"] = "completed"
                risk["disease"] = self.name
                risk.setdefault("prediction_method", "conversational_agent")
                risk.setdefault("method", risk["prediction_method"])
                risk.setdefault("risk_factors", [])
                risk.setdefault("recommendations", [])
                risk.setdefault("summary", risk.get("response", ""))
                return risk
            error = risk.get("error") or "Risk predictor returned an unsuccessful result"
            fallback = self.rule_risk(data, analysis)
            fallback["fallback_reason"] = error
            logger.warning("RiskPredictorAgent returned an error; using rule-based fallback: %s", error)
            return fallback
        except Exception as e:
            logger.warning("RiskPredictorAgent failed; using rule-based fallback: %s", e)

            fallback = self.rule_risk(data, analysis)
            fallback["fallback_reason"] = str(e)
            return fallback

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

    # ── shared risk explanation ───────────────────────────────────────────

    def risk_factors(self, data: Dict[str, Any]) -> tuple:
        """Derive the clinical risk factors and recommendations for a case.

        This is the single source of the diabetes risk narrative: the plugin,
        the report renderer and the alert agent all read it from here.
        """
        risk_factors = []
        recommendations = []
        hba1c = _as_float(data.get("hba1c"))
        glucose = _as_float(data.get("glucose"))
        bmi = _as_float(data.get("bmi"))
        age = _as_float(data.get("age"))

        if hba1c is not None and hba1c >= 6.5:
            risk_factors.append({"parameter": "hba1c", "value": hba1c, "impact": "High"})
            recommendations.append("Consult endocrinologist for diabetes management")
        elif hba1c is not None and hba1c >= 5.7:
            risk_factors.append({"parameter": "hba1c", "value": hba1c, "impact": "Medium"})
            recommendations.append("Monitor HbA1c every 3 months")

        if glucose is not None and glucose >= 126:
            risk_factors.append({"parameter": "glucose", "value": glucose, "impact": "High"})
            recommendations.append("Regular fasting glucose monitoring recommended")
        elif glucose is not None and glucose >= 100:
            risk_factors.append({"parameter": "glucose", "value": glucose, "impact": "Medium"})

        if bmi is not None and bmi >= 25:
            risk_factors.append({
                "parameter": "bmi",
                "value": bmi,
                "impact": "High" if bmi >= 30 else "Medium",
            })
            recommendations.append("Weight management program recommended")

        if age is not None and age >= 45:
            risk_factors.append({"parameter": "age", "value": age, "impact": "Medium"})

        if not recommendations:
            recommendations.append("Maintain healthy lifestyle and regular checkups")

        return risk_factors, recommendations


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
