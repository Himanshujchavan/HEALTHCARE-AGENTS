from typing import List, Dict, Any, Optional
from Agents.plugins import registry

class DifferentialAgent:
    """Analyzes overlapping symptoms across multiple suspected diseases."""

    @staticmethod
    def run_differential(disease_results: Dict[str, Dict], symptoms: List[str]) -> Optional[Dict[str, Any]]:
        if not disease_results:
            return None

        # Filter to actual registered disease plugins
        valid_diseases = {
            name: data for name, data in disease_results.items()
            if registry.get(name) is not None
        }

        if len(valid_diseases) <= 1:
            return None

        scored_diseases = []
        user_symptoms_lower = [s.lower() for s in (symptoms or [])]

        for name, result in valid_diseases.items():
            risk_info = result.get("risk") or {}
            symptom_info = result.get("symptom") or {}

            # Extract risk probability or score in [0, 1]
            risk_prob = risk_info.get("risk_probability")
            if risk_prob is None:
                risk_score = risk_info.get("score")
                risk_prob = (risk_score / 100.0) if risk_score is not None else 0.0

            # Extract symptom severity in [0, 1]
            sym_score = symptom_info.get("severity_score")
            if sym_score is None:
                sym_score = symptom_info.get("severity", 0.0)

            try:
                risk_prob = float(risk_prob)
            except (ValueError, TypeError):
                risk_prob = 0.0

            try:
                sym_score = float(sym_score)
            except (ValueError, TypeError):
                sym_score = 0.0

            # Rank by risk score (60%) plus symptom overlap (40%)
            combined_score = round((risk_prob * 0.6) + (sym_score * 0.4), 4)
            scored_diseases.append({
                "disease": name,
                "score": combined_score,
                "risk_probability": risk_prob,
                "symptom_score": sym_score
            })

        # Sort by score descending
        scored_diseases.sort(key=lambda x: x["score"], reverse=True)

        # Calculate shared symptoms across the plugins
        disease_matched = {}
        for name in valid_diseases.keys():
            plugin = registry.get(name)
            matched = set()
            if plugin and hasattr(plugin, "symptom_profiles"):
                primary_profile = plugin.symptom_profiles.get("primary", [])
                for s in user_symptoms_lower:
                    if any(p in s for p in primary_profile):
                        matched.add(s)
            disease_matched[name] = matched

        if len(disease_matched) >= 2:
            shared_set = set.intersection(*disease_matched.values()) if disease_matched else set()
            shared = sorted(list(shared_set))
        else:
            shared = []

        return {
            "primary": scored_diseases[0]["disease"],
            "alternatives": scored_diseases[1:],
            "shared_symptoms": shared
        }
