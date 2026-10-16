from typing import List, Dict, Any, Optional
from .base import DiseasePlugin
import numpy as np

class DiabetesPlugin(DiseasePlugin):
    name = "diabetes"
    required_fields = ["hba1c", "glucose", "bmi", "age"]
    optional_fields = ["blood_pressure", "creatinine"]

    reference_ranges = {
        "hba1c": {"low": 4.0, "high": 5.6, "unit": "%"},
        "glucose": {"low": 70, "high": 99, "unit": "mg/dL"},
        "bmi": {"low": 18.5, "high": 24.9, "unit": "kg/m2"},
        "age": {"low": 0, "high": 120, "unit": "years"},
    }

    trend_scales = {
        "hba1c": 0.5,
        "glucose": 20.0,
        "bmi": 2.0
    }

    symptom_profiles = {
        "primary": ["polyuria", "polydipsia", "polyphagia", "unexplained weight loss"],
        "complications": {
            "cardiovascular": ["chest pain", "shortness of breath", "palpitations"],
            "cerebrovascular": ["slurred speech", "face droop", "one-side weakness"],
            "renal": ["foamy urine", "reduced urine output", "swelling in ankles"],
            "retinopathy": ["blurred vision", "dark spots", "vision loss"]
        }
    }

    alert_thresholds = {
        "hba1c": 8.0,
        "glucose": 200.0
    }

    def flags(self, data: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
        flags = []
        hba1c = data.get("hba1c", 0)
        if hba1c >= 6.5:
            flags.append("DIABETES CRITERIA MET")
        elif hba1c >= 5.7:
            flags.append("PRE-DIABETES RANGE")
        return flags

    def note(self, param: str, status: str) -> str:
        notes_map = {
            "hba1c": {"High": "Diabetic range (ADA 6.5%+)", "Low": "Hypoglycemia risk"},
            "glucose": {"High": "Hyperglycemic state", "Low": "Low blood sugar"},
            "bmi": {"High": "Overweight/Obese range", "Low": "Underweight range"},
        }
        return notes_map.get(param, {}).get(status, "Check clinical guidelines")

    def notes(self, param: str, status: str) -> str:
        return self.note(param, status)

    def rule_risk(self, data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        # Migrated from riskpredictor.py
        hba1c = data.get("hba1c", 0)
        glucose = data.get("glucose", 0)
        bmi = data.get("bmi", 0)
        age = data.get("age", 0)

        # Scores based on migration map
        hba1c_score = 1.0 if hba1c >= 6.5 else (0.6 if hba1c >= 5.7 else 0.0)
        glucose_score = 1.0 if glucose >= 126 else (0.6 if glucose >= 100 else 0.0)

        bmi_score = 0.0
        if bmi >= 40: bmi_score = 1.0
        elif bmi >= 35: bmi_score = 0.85
        elif bmi >= 30: bmi_score = 0.7
        elif bmi >= 25: bmi_score = 0.4

        age_score = 0.0
        if age >= 65: age_score = 0.8
        elif age >= 55: age_score = 0.6
        elif age >= 45: age_score = 0.4
        elif age >= 35: age_score = 0.2

        score = (hba1c_score * 0.35) + (glucose_score * 0.35) + (bmi_score * 0.20) + (age_score * 0.10)
        final_score = score * 100

        level = "LOW"
        if final_score > 60: level = "CRITICAL"
        elif final_score > 40: level = "HIGH"
        elif final_score > 20: level = "MEDIUM"

        return {"score": final_score, "level": level, "method": "rule-based"}

    def predict_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Fallback to rule_risk if Pima model not available or data insufficient
        # In a real scenario, this would load PimaModel.pkl
        return self.rule_risk(data, {})

    def score_symptoms(self, symptoms: List[str]) -> Dict[str, Any]:
        # Migrated from symptomchecker.py
        match_count = 0
        primary_symptoms = self.symptom_profiles["primary"]
        for s in symptoms:
            if any(ps in s.lower() for ps in primary_symptoms):
                match_count += 1

        severity = min(match_count / max(len(primary_symptoms), 1), 1.0)
        return {"severity": severity, "top_hypothesis": "Diabetes Mellitus" if severity > 0.3 else "Unknown"}
