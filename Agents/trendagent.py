from typing import List, Dict, Any
from .plugins.base import DiseasePlugin

class TrendAgent:
    """Analyzes health trends over time for a specific disease."""

    @staticmethod
    def run_trend(history: List[Dict], plugin: DiseasePlugin, current: Dict) -> Dict[str, Any]:
        if not history or len(history) < 2:
            return {
                "trend_score": 0.0,
                "direction": "stable",
                "visits_used": 0,
                "status": "insufficient_history"
            }

        max_delta_score = 0.0
        direction = "stable"

        # Analyze parameters defined in plugin.trend_scales
        for param, scale in plugin.trend_scales.items():
            if param not in current:
                continue

            # Get the most recent historical value
            past_values = [h.get("analysis", {}).get("parameters", {}).get(param)
                          for h in history if "analysis" in h and param in h["analysis"]["parameters"]]

            if not past_values:
                continue

            last_val = past_values[-1]
            if last_val is None: continue

            current_val = current.get(param)
            if current_val is None: continue

            delta = current_val - last_val
            # Score is clamp(delta / scale, 0, 1)
            param_score = max(0.0, min(1.0, delta / scale))

            if param_score > max_delta_score:
                max_delta_score = param_score
                direction = "rising" if delta > 0 else "falling"

        return {
            "trend_score": max_delta_score,
            "direction": direction,
            "visits_used": len(history),
            "status": "completed"
        }
