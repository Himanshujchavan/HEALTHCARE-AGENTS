from typing import List, Dict, Any, Optional

class DifferentialAgent:
    """Analyzes overlapping symptoms across multiple suspected diseases."""

    @staticmethod
    def run_differential(disease_results: Dict[str, Dict], symptoms: List[str]) -> Optional[Dict[str, Any]]:
        if not disease_results:
            return None

        # Only run if more than one disease is applicable
        if len(disease_results) <= 1:
            return None

        scored_diseases = []
        for name, result in disease_results.items():
            # Use symptom severity as the primary metric for differential
            score = result.get("symptoms", {}).get("severity", 0.0)
            scored_diseases.append({"disease": name, "score": score})

        # Sort by score descending
        scored_diseases.sort(key=lambda x: x["score"], reverse=True)

        # Find shared symptoms (simplified)
        # In a full version, this would intersect symptom profiles
        shared = []
        # Just a placeholder for the intersection logic
        # shared = set(symptoms).intersection(plugin1.profiles).intersection(plugin2.profiles)

        return {
            "primary": scored_diseases[0]["disease"],
            "alternatives": scored_diseases[1:],
            "shared_symptoms": shared
        }
