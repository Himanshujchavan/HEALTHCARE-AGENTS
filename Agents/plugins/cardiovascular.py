"""Cardiovascular plugin.

Provides reference ranges, clinical flags, a deterministic rule scorer, an
ML-backed predictor (logistic regression over the cardiovascular disease
dataset) and symptom scoring for cardiovascular risk. Emits the same canonical
risk contract as every other plugin.

Two conventions matter here:

* **An unavailable feature is not a negative value.** Smoking status is
  optional: when it is absent the rule scorer drops the smoking component and
  redistributes its weight, and says so, rather than reading "not supplied"
  as "does not smoke".
* **The ML model's label is a disease likelihood, not a time-to-event
  estimate.** The model is trained on a cross-sectional dataset, so its
  probability is not a 10-year risk. The prediction carries that qualification
  and the model's own confidence and limitations.
"""

from typing import List, Dict, Any, Optional
import logging
from .base import DiseasePlugin
from utils.helpers import normalize_level

logger = logging.getLogger(__name__)

#: Weights of the rule scorer's components. The smoking component is removed
#: and its weight redistributed when the smoking status is unknown.
RULE_WEIGHTS = {
    "age": 0.2,
    "systolic_bp": 0.3,
    "total_cholesterol": 0.3,
    "smoker": 0.2,
}


class CardiovascularPlugin(DiseasePlugin):
    name = "cardiovascular"
    # Identical to the ML model's base feature set: a case the plugin accepts
    # can always be scored by the model, and a case the plugin rejects cannot.
    required_fields = ["age", "systolic_bp", "total_cholesterol"]
    optional_fields = [
        "smoker", "diastolic_bp", "ldl", "hdl", "glucose", "bmi",
        "chest_pain_type", "st_slope",
    ]

    reference_ranges = {
        "systolic_bp": {"low": 90, "high": 120, "unit": "mmHg"},
        "diastolic_bp": {"low": 60, "high": 80, "unit": "mmHg"},
        "total_cholesterol": {"low": 0, "high": 200, "unit": "mg/dL"},
        "ldl": {"low": 0, "high": 100, "unit": "mg/dL"},
        "hdl": {"low": 40, "high": 100, "unit": "mg/dL"},
        # Shared metabolic parameters: evaluated here too so a cardiovascular
        # analysis never reports them as "Unknown" when they were submitted.
        "glucose": {"low": 70, "high": 99, "unit": "mg/dL"},
        "bmi": {"low": 18.5, "high": 24.9, "unit": "kg/m2"},
        # Ordinal categories. These are reported, not ranged — a code is not a
        # measurement, so "Normal"/"High" would be meaningless.
        "chest_pain_type": {"low": 0, "high": 3, "unit": "category", "categorical": True},
        "st_slope": {"low": 0, "high": 3, "unit": "category", "categorical": True},
    }

    #: Human-readable meaning of the ordinal category codes.
    #:
    #: The clinical meaning of each code could not be verified for this copy
    #: of the dataset: its ``slope`` column has four levels where the upstream
    #: UCI variable has three, so the codes are not a simple re-encoding of
    #: the published categories. They are reported as the dataset's own
    #: ordinal codes rather than under invented clinical names.
    category_labels = {
        "chest_pain_type": {
            0: "Chest pain category 0 (lowest risk band in the source dataset)",
            1: "Chest pain category 1",
            2: "Chest pain category 2 (highest risk band in the source dataset)",
            3: "Chest pain category 3 (highest risk band in the source dataset)",
        },
        "st_slope": {
            0: "ST segment category 0 (lowest risk band in the source dataset)",
            1: "ST segment category 1",
            2: "ST segment category 2 (highest risk band in the source dataset)",
            3: "ST segment category 3 (highest risk band in the source dataset)",
        },
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

        ldl = data.get("ldl")
        if ldl is not None:
            try:
                if float(ldl) >= 100:
                    flags.append("HIGH LDL")
            except (ValueError, TypeError):
                pass

        if data.get("smoker") is True:
            flags.append("SMOKING RISK FACTOR")
        elif data.get("smoker") is None:
            flags.append("SMOKING STATUS UNKNOWN")

        return flags

    def notes(self, param: str, status: str) -> str:
        notes_map = {
            "systolic_bp": {"High": "Hypertensive range", "Low": "Hypotension risk"},
            "diastolic_bp": {"High": "Elevated diastolic pressure", "Low": "Hypotension risk"},
            "total_cholesterol": {"High": "Hyperlipidemia risk", "Low": "Normal"},
            "ldl": {"High": "High LDL risk", "Low": "Optimal LDL"},
            "hdl": {"High": "Protective HDL", "Low": "Low HDL risk factor"},
            "glucose": {"High": "Elevated fasting glucose", "Low": "Low blood sugar"},
            "bmi": {"High": "Overweight/Obese range", "Low": "Underweight range"},
            "smoker": {"Reported": "Current smoking status", "High": "Current smoker"},
            "chest_pain_type": {"Reported": "Chest pain type"},
            "st_slope": {"Reported": "Peak-exercise ST segment slope"},
        }
        return notes_map.get(param, {}).get(status, "Check clinical guidelines")

    def category_label(self, param: str, value: Any) -> Optional[str]:
        """The human-readable meaning of an ordinal category code."""
        labels = self.category_labels.get(param, {})
        try:
            return labels.get(int(value))
        except (TypeError, ValueError):
            return None

    def rule_risk(self, data: Dict[str, Any], analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        age = float(data.get("age", 0) or 0)
        sbp = float(data.get("systolic_bp", 0) or 0)
        chol = float(data.get("total_cholesterol", 0) or 0)

        # Smoking status has three states, not two: current smoker, confirmed
        # non-smoker, and unknown. An unknown status is not a negative, so the
        # smoking component is dropped and its weight redistributed instead of
        # contributing a silent zero.
        smoker_value = data.get("smoker")
        smoker_known = smoker_value is not None

        age_score = 1.0 if age >= 60 else (0.6 if age >= 45 else 0.2)
        sbp_score = 1.0 if sbp >= 160 else (0.6 if sbp >= 140 else 0.2)
        chol_score = 1.0 if chol >= 240 else (0.6 if chol >= 200 else 0.2)
        smoker_score = 1.0 if smoker_value is True else 0.0

        components = {
            "age": age_score,
            "systolic_bp": sbp_score,
            "total_cholesterol": chol_score,
        }
        weights = {k: RULE_WEIGHTS[k] for k in components}
        if smoker_known:
            components["smoker"] = smoker_score
            weights["smoker"] = RULE_WEIGHTS["smoker"]
        else:
            # Redistribute the smoking weight across the components that are
            # actually known.
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}

        score = sum(components[key] * weights[key] for key in components)
        final_score = round(score * 100, 2)

        level = "Low"
        if final_score > 60:
            level = "Critical"
        elif final_score > 40:
            level = "High"
        elif final_score > 20:
            level = "Moderate"

        level = normalize_level(level)
        risk_factors, recommendations = self.risk_factors(data)

        assumptions = []
        if not smoker_known:
            assumptions.append(
                "Smoking status was not supplied, so the smoking component was "
                "excluded from the score and its weight redistributed across "
                "the known components."
            )

        return {
            "status": "completed",
            "disease": self.name,
            "score": final_score,
            "level": level,
            "risk_level": level,
            "risk_probability": round(final_score / 100.0, 2),
            "risk_score": final_score,
            "risk_percentage": f"{int(round(final_score))}%",
            "method": "rule_based",
            "prediction_method": "rule_based",
            "smoker_status": (
                "current" if smoker_value is True
                else "non_smoker" if smoker_value is False
                else "unknown"
            ),
            "assumptions": assumptions,
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "summary": (
                f"Rule-based cardiovascular risk assessment: {level} "
                f"(Score: {final_score}%)."
            )
        }

    def predict_risk(self, data: Dict[str, Any], analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            from .cardio_model import CardioModel
            model = CardioModel()
            prediction = model.predict(data)

            if prediction is not None:
                prob = prediction["probability"]
                variant = prediction["variant"]
                missing_optional = prediction["missing_optional_features"]

                level = "Low"
                if prob > 0.7:
                    level = "Critical"
                elif prob > 0.4:
                    level = "High"
                elif prob > 0.2:
                    level = "Moderate"

                level = normalize_level(level)
                risk_factors, recommendations = self.risk_factors(data)
                metadata = model.metadata()
                variant_metrics = (metadata.get("metrics") or {}).get(variant, {})
                confidence = variant_metrics.get("confidence", metadata.get("confidence", "unknown"))
                target_semantics = metadata.get("target_semantics", "")
                limitations = metadata.get("limitations", [])

                if variant == "extended":
                    variant_note = (
                        "Scored with the extended model, which also uses the "
                        "chest pain type and ST slope."
                    )
                else:
                    variant_note = (
                        "Scored with the base model"
                        + (
                            f"; {', '.join(missing_optional)} "
                            f"{'were' if len(missing_optional) > 1 else 'was'} "
                            "not supplied, so the extended model was not used."
                            if missing_optional
                            else "."
                        )
                    )

                return {
                    "status": "completed",
                    "disease": self.name,
                    "score": round(prob * 100, 2),
                    "level": level,
                    "risk_level": level,
                    "risk_probability": round(prob, 2),
                    "risk_score": round(prob * 100, 2),
                    "risk_percentage": f"{int(round(prob * 100))}%",
                    "method": "ML-based (Logistic Regression)",
                    "prediction_method": "ML-based (Logistic Regression)",
                    "model_variant": variant,
                    "model_features": prediction["features"],
                    "model_missing_optional_features": missing_optional,
                    "model_metrics": variant_metrics,
                    "risk_factors": risk_factors,
                    "recommendations": recommendations,
                    "model_metadata": metadata,
                    "model_confidence": confidence,
                    "model_target": target_semantics,
                    "model_limitations": limitations,
                    "disclaimer": (
                        "Estimated probability of cardiovascular disease, not a "
                        "10-year or other time-to-event risk estimate."
                    ),
                    "summary": (
                        f"ML-based cardiovascular assessment: {level} "
                        f"(estimated probability of cardiovascular disease "
                        f"{prob:.2f}). This is a disease likelihood from a "
                        f"model with {confidence} confidence, not a 10-year "
                        f"risk estimate. {variant_note}"
                    ),
                }
        except Exception as e:
            logger.warning(f"Cardio ML Prediction failed, falling back to rules: {e}")
            fallback = self.rule_risk(data, analysis)
            fallback["fallback_reason"] = str(e)
            return fallback

        fallback = self.rule_risk(data, analysis)
        fallback["fallback_reason"] = (
            "The cardiovascular model could not score this case; the "
            "rule-based score was used instead."
        )
        return fallback

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

    # ── shared risk explanation ───────────────────────────────────────────

    def risk_factors(self, data: Dict[str, Any]) -> tuple:
        """Derive the clinical risk factors and recommendations for a case."""
        risk_factors = []
        recommendations = []
        age = _as_float(data.get("age"))
        sbp = _as_float(data.get("systolic_bp"))
        chol = _as_float(data.get("total_cholesterol"))
        ldl = _as_float(data.get("ldl"))
        hdl = _as_float(data.get("hdl"))

        if age is not None and age >= 45:
            risk_factors.append({
                "parameter": "age",
                "value": age,
                "impact": "High" if age >= 60 else "Medium",
            })

        if sbp is not None and sbp >= 140:
            risk_factors.append({"parameter": "systolic_bp", "value": sbp, "impact": "High"})
            recommendations.append("Blood pressure review with a clinician")
        elif sbp is not None and sbp >= 120:
            risk_factors.append({"parameter": "systolic_bp", "value": sbp, "impact": "Medium"})

        if chol is not None and chol >= 240:
            risk_factors.append({"parameter": "total_cholesterol", "value": chol, "impact": "High"})
            recommendations.append("Discuss lipid-lowering options with a clinician")
        elif chol is not None and chol >= 200:
            risk_factors.append({"parameter": "total_cholesterol", "value": chol, "impact": "Medium"})

        if ldl is not None and ldl >= 100:
            risk_factors.append({"parameter": "ldl", "value": ldl, "impact": "High"})
        if hdl is not None and hdl < 40:
            risk_factors.append({"parameter": "hdl", "value": hdl, "impact": "Medium"})

        smoker_value = data.get("smoker")
        if smoker_value is True:
            risk_factors.append({"parameter": "smoker", "value": True, "impact": "High"})
            recommendations.append("Smoking cessation support is recommended")
        elif smoker_value is None:
            # Not supplied is not the same as "does not smoke". Say so rather
            # than scoring it as a genuine negative.
            risk_factors.append(
                {"parameter": "smoker", "value": None, "impact": "Unknown"}
            )
            recommendations.append(
                "Confirm smoking status — it was not provided and the score "
                "does not account for it"
            )

        # The ordinal categories are reported, not scored: their clinical
        # meaning could not be verified for this dataset copy, so they inform
        # the model but carry no rule weight here.
        for param in ("chest_pain_type", "st_slope"):
            value = data.get(param)
            if value is None:
                continue
            risk_factors.append({
                "parameter": param,
                "value": value,
                "impact": "Reported",
                "label": self.category_label(param, value),
            })

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
