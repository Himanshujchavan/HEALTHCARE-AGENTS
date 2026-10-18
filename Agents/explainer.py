from typing import Dict, List, Any, Optional

class Explainer:
    """Generates a clinical explanation for the final risk assessment."""

    @staticmethod
    def explain(
        final_assessment: Dict[str, Any],
        trend: Optional[Dict[str, Any]] = None,
        triggers: Optional[List[str]] = None
    ) -> str:
        if not final_assessment:
            return "No assessment data available to generate explanation."

        disease = final_assessment.get("disease", "Condition").title()
        score = final_assessment.get("score", 0.0)
        level = final_assessment.get("risk_level") or final_assessment.get("level", "Unknown")
        formula = final_assessment.get("formula", "Standard Clinical Multi-Modal Fusion")

        # Format percentage: if score is 0-1, multiply by 100
        score_pct = (score * 100) if score <= 1.0 else score

        lines = [
            f"Assessment for {disease}: Risk level is {level} (Estimated Risk Score: {score_pct:.1f}%).",
            f"Formula applied: {formula}."
        ]

        components = final_assessment.get("components")
        if isinstance(components, dict):
            drivers = []
            if "ml_component" in components:
                drivers.append(f"Model Risk: {components['ml_component']}")
            if "abnormal_component" in components:
                drivers.append(f"Abnormal Labs: {components['abnormal_component']}")
            if "symptom_component" in components:
                drivers.append(f"Symptoms: {components['symptom_component']}")
            if "trend_component" in components and components["trend_component"] > 0:
                drivers.append(f"Trend Trajectory: {components['trend_component']}")

            if drivers:
                lines.append("Contributing Components: " + ", ".join(drivers) + ".")

        abnormal_count = final_assessment.get("abnormal_count")
        if abnormal_count:
            lines.append(f"Identified {abnormal_count} abnormal lab parameters.")

        if triggers:
            lines.append("Alert Triggers:\n- " + "\n- ".join(triggers))

        if trend and trend.get("status") == "completed":
            direction = trend.get("direction", "stable")
            visits = trend.get("visits_used", 2)
            lines.append(f"Trend Trajectory: Biomarkers exhibit a {direction} trajectory across {visits} recorded visits.")
            if trend.get("rising_but_normal"):
                lines.append(f"Parameters rising within normal boundaries: {', '.join(trend['rising_but_normal'])}.")
        elif trend and trend.get("status") == "insufficient_history":
            lines.append("Trend Analysis: Baseline visit; longitudinal trajectory will be tracked across future visits.")

        return "\n".join(lines)
