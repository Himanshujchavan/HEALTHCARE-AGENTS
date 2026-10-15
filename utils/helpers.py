def normalize_level(level: str) -> str:
    """
    Unifies risk levels across different agents.
    Maps:
    - MEDIUM -> Moderate
    - LOW -> Low
    - HIGH -> High
    - CRITICAL -> Critical
    - Very Low -> Low
    - Very High -> Critical
    """
    if not level:
        return "Unknown"

    mapping = {
        "MEDIUM": "Moderate",
        "Moderate": "Moderate",
        "LOW": "Low",
        "Low": "Low",
        "Very Low": "Low",
        "HIGH": "High",
        "High": "High",
        "CRITICAL": "Critical",
        "Critical": "Critical",
        "Very High": "Critical",
    }
    return mapping.get(level, level)
