from typing import List, Dict, Any, Optional

def run_trend(history: List[Dict], plugin: Any, current: Dict) -> Dict[str, Any]:
    """
    Calculate trend score based on historical records.
    Returns: {trend_score, direction, rising_but_normal, visits_used}
    """
    ...

def run_differential(disease_results: Dict[str, Any], symptoms: List[str]) -> Dict[str, Any]:
    """
    Analyze symptoms across applicable diseases to find the primary match.
    Returns: {primary, alternatives: [{disease, score}], shared_symptoms}
    """
    ...

def explain(final_assessment: Dict[str, Any], trend: Optional[Dict], triggers: List[str]) -> str:
    """
    Generate a human-readable explanation of the risk score.
    """
    ...
