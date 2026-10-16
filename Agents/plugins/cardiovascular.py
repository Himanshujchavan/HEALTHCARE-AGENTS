from typing import List, Dict, Any, Optional
from .base import DiseasePlugin
import numpy as np

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
        "total_cholesterol": 15.0
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
        sbp = data.get("systolic_bp", 0)
        if sbp >= 140:
            flags.append("HYPERTENSION RANGE")
        chol = data.get("total_cholesterol", 0)
        if chol >= 240:
            flags.append("HIGH LDL/CHOLESTEROL")
        return flags

    def notes(self, param: str, status: str) -> str:
        notes_map = {
            "systolic_bp": {"High": "Hypertensive range", "Low": "Hypotension risk"},
            "total_cholesterol": {"High": "Hyperlipidemia risk", "Low": "Normal"},
        }
        return notes_map.get(param, {}).get(status, "Check clinical guidelines")

    def rule_risk(self, data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        # Simplified cardiovascular rule-based risk
        age = data.get("age", 0)
        sbp = data.get("systolic_bp", 0)
        chol = data.get("total_cholesterol", 0)
        smoker = 1 if data.get("smoker") else 0

        # Scoring logic
        age_score = 1.0 if age >= 60 else (0.6 if age >= 45 else 0.2)
        sbp_score = 1.0 if sbp >= 160 else (0.6 if sbp >= 140 else 0.2)
        chol_score = 1.0 if chol >= 240 else (0.6 if chol >= 200 else 0.2)
        smoker_score = 1.0 if smoker else 0.0

        score = (age_score * 0.2) + (sbp_score * 0.3) + (chol_score * 0.3) + (smoker_score * 0.2)
        final_score = score * 100

        level = "LOW"
        if final_score > 60: level = "CRITICAL"
        elif final_score > 40: level = "HIGH"
        elif final_score > 20: level = "MEDIUM"

        return {"score": final_score, "level": level, "method": "rule-based"}

    def predict_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Attempt ML prediction first
        try:
            from .cardio_model import CardioModel
            model = CardioModel()
            prob = model.predict_probability(data)

            if prob is not None:
                # Map probability to risk level
                level = "LOW"
                if prob > 0.7: level = "CRITICAL"
                elif prob > 0.4: level = "HIGH"
                elif prob > 0.2: level = "MEDIUM"

                return {
                    "score": round(prob * 100, 2),
                    "risk_level": level,
                    "risk_probability": prob,
                    "method": "ML-based (Logistic Regression)"
                }
        except Exception as e:
            print(f"ML Prediction failed, falling back to rules: {e}")

        # Fallback to rule-based risk
        return self.rule_risk(data, {})

    def score_symptoms(self, symptoms: List[str]) -> Dict[str, Any]:
        match_count = 0
        primary = self.symptom_profiles["primary"]
        for s in symptoms:
            if any(ps in s.lower() for ps in primary):
                match_count += 1

        severity = min(match_count / max(len(primary), 1), 1.0)
        return {"severity": severity, "top_hypothesis": "Cardiovascular Disease" if severity > 0.3 else "Unknown"}
