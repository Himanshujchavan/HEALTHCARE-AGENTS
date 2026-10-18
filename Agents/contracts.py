"""Data contracts and functional entry points for sub-agents.

History Record Schema:
    {
        "created_at": str (ISO 8601),
        "params": Dict[str, float] (e.g. {"hba1c": 6.8, "glucose": 148, ...}),
        "risk_score": float (final assessment score 0-1 or 0-100)
    }

Disease Results Schema:
    {
        "<disease_name>": {
            "analysis": Dict[str, Any],
            "risk": Dict[str, Any],
            "symptom": Dict[str, Any],
            "final_assessment": Dict[str, Any]
        }
    }
"""

from typing import List, Dict, Any, Optional
from Agents.trendagent import TrendAgent
from Agents.differentialagent import DifferentialAgent
from Agents.explainer import Explainer

def run_trend(history: List[Dict], plugin: Any, current: Dict) -> Dict[str, Any]:
    """Calculate trend score based on historical records.
    Returns: {trend_score, direction, rising_but_normal, visits_used, status}
    """
    return TrendAgent.run_trend(history, plugin, current)

def run_differential(disease_results: Dict[str, Any], symptoms: List[str]) -> Optional[Dict[str, Any]]:
    """Analyze symptoms across applicable diseases to find the primary match.
    Returns: {primary, alternatives: [{disease, score}], shared_symptoms}
    """
    return DifferentialAgent.run_differential(disease_results, symptoms)

def explain(final_assessment: Dict[str, Any], trend: Optional[Dict] = None, triggers: Optional[List[str]] = None) -> str:
    """Generate a human-readable explanation of the risk score."""
    return Explainer.explain(final_assessment, trend, triggers)
