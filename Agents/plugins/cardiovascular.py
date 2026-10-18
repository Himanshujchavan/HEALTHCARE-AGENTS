from typing import List, Dict, Any, Optional
import logging
from .base import DiseasePlugin
from utils.helpers import normalize_level

logger = logging.getLogger(__name__)

class CardiovascularPlugin(DiseasePlugin):
    name = "cardiovascular"
    required_fields = ["age", "systolic_bp", "total_cholesterol", "smoker"]
    optional_fields = ["diastolic_bp", "ldl", "hdl", "glucose", "bmi"]

    reference_ranges = {
        "systolic_bp": {"low": 90, "high": 120, "unit": "mmHg"},
        "diastolic_bp": {"low": 60, "high": 80, "unit": "mmHg"},
        "total_cholesterol": {"low": 0, "high": 200, "unit": "mg/dL"},
        "ldl": {"low": 0, "high": 100, "unit": "mg/dL"},
        "hdl": {"low": 40, "high": 100, "unit": "mg/dL"},
    }

    trend_scales = {
        "systolic_bp": 10.0,
        "total_cholesterol": 15.0,
        "ldl": 10.0,
        "hdl": -10.0,
    }

    symptom_profiles = {
        "primary": ["chest pain", "shortness of breath", "palpitations", "dizziness", "fatigue"],
        "red_flags": ["crushing chest pain", "left-arm pain", "jaw pain", "sudden fainting"]
    }

    alert_thresholds = {
        "systolic_bp": 160.0,
        "total_cholesterol": 240.0
    }

    def flags(self, data: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
        flags = []
        sbp = data.get("systolic_bp")
        if sbp is not None:
            try:
                if float(sbp) >= 140:
                    flags.append("HYPERTENSION RANGE")
            except (ValueError, TypeError):
                pass

        chol = data.get("total_cholesterol")
        if chol is not None:
            try:
                if float(chol) >= 240:
                    flags.append("HIGH LDL/CHOLESTEROL")
            except (ValueError, TypeError):
                pass
        return flags

    def notes(self, param: str, status: str) -> str:
        notes_map = {
            "systolic_bp": {"High": "Hypertensive range", "Low": "Hypotension risk"},
            "diastolic_bp": {"High": "Elevated diastolic pressure", "Low": "Hypotension risk"},
            "total_cholesterol": {"High": "Hyperlipidemia risk", "Low": "Normal"},
            "ldl": {"High": "High LDL risk", "Low": "Optimal LDL"},
            "hdl": {"High": "Protective HDL", "Low": "Low HDL risk factor"},
        }
        return notes_map.get(param, {}).get(status, "Check clinical guidelines")

    def rule_risk(self, data: Dict[str, Any], analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        age = float(data.get("age", 0) or 0)
        sbp = float(data.get("systolic_bp", 0) or 0)
        chol = float(data.get("total_cholesterol", 0) or 0)
        smoker = 1 if data.get("smoker") else 0

        age_score = 1.0 if age >= 60 else (0.6 if age >= 45 else 0.2)
        sbp_score = 1.0 if sbp >= 160 else (0.6 if sbp >= 140 else 0.2)
        chol_score = 1.0 if chol >= 240 else (0.6 if chol >= 200 else 0.2)
        smoker_score = 1.0 if smoker else 0.0

        score = (age_score * 0.2) + (sbp_score * 0.3) + (chol_score * 0.3) + (smoker_score * 0.2)
        final_score = round(score * 100, 2)

        level = "Low"
        if final_score > 60:
            level = "Critical"
        elif final_score > 40:
            level = "High"
        elif final_score > 20:
            level = "Moderate"

        level = normalize_level(level)

        return {
            "score": final_score,
            "level": level,
            "risk_level": level,
            "risk_probability": round(final_score / 100.0, 2),
            "risk_percentage": f"{int(round(final_score))}%",
            "method": "rule-based"
        }

    def predict_risk(self, data: Dict[str, Any], analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            from .cardio_model import CardioModel
            model = CardioModel()
            prob = model.predict_probability(data)

            if prob is not None:
                level = "Low"
                if prob > 0.7:
                    level = "Critical"
                elif prob > 0.4:
                    level = "High"
                elif prob > 0.2:
                    level = "Moderate"

                level = normalize_level(level)

                return {
                    "score": round(prob * 100, 2),
                    "level": level,
                    "risk_level": level,
                    "risk_probability": round(prob, 2),
                    "risk_percentage": f"{int(round(prob * 100))}%",
                    "method": "ML-based (Logistic Regression)"
                }
        except Exception as e:
            logger.warning(f"Cardio ML Prediction failed, falling back to rules: {e}")

        return self.rule_risk(data, analysis)

    def score_symptoms(self, symptoms: List[str]) -> Dict[str, Any]:
        match_count = 0
        primary = self.symptom_profiles["primary"]
        red_flags = self.symptom_profiles.get("red_flags", [])

        detected_red_flags = []
        for s in symptoms:
            s_lower = s.lower()
            if any(rf in s_lower for rf in red_flags):
                detected_red_flags.append(s)
            if any(ps in s_lower for ps in primary):
                match_count += 1

        severity = min(match_count / max(len(primary), 1), 1.0)
        has_emergency = len(detected_red_flags) > 0
        if has_emergency:
            severity = max(severity, 1.0)

        return {
            "severity": severity,
            "severity_score": severity,
            "top_hypothesis": "Cardiovascular Disease" if (severity > 0.3 or has_emergency) else "Unknown",
            "emergency": has_emergency,
            "red_flags": detected_red_flags
        }
