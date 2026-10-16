from typing import Dict, List, Any

class Explainer:
    """Generates a clinical explanation for the final risk assessment."""

    @staticmethod
    def explain(final_assessment: Dict, trend: Dict, triggers: List[str]) -> str:
        disease = final_assessment.get("disease", "Unknown Condition")
        score = final_assessment.get("score", 0.0)
        level = final_assessment.get("level", "Unknown")
        formula = final_assessment.get("formula", "Standard")

        explanation = f"Assessment for {disease}: Risk level is {level} (Score: {score:.1f}%).\n"
        explanation += f"Formula used: {formula}.\n\n"

        if triggers:
            explanation += "Key Triggers:\n- " + "\n- ".join(triggers) + "\n"

        if trend and trend.get("status") == "completed":
            dir = trend.get("direction", "stable")
            explanation += f"\nTrend Analysis: Health markers are {dir} compared to previous visits."
        elif trend and trend.get("status") == "insufficient_history":
            explanation += "\nTrend Analysis: Insufficient history for trajectory mapping."

        return explanation
