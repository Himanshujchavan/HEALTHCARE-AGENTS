from typing import Protocol, List, Dict, Any

class DiseasePlugin:
    name: str
    required_fields: List[str]
    optional_fields: List[str]
    reference_ranges: Dict[str, Any]
    param_alert_thresholds: Dict[str, float]      # e.g. {"hba1c": 7.0}
    trend_scales: Dict[str, float]                # param -> delta counting as a full trend
    symptom_profiles: Dict[str, Any]

    def flags(self, data: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
        """Return a list of flags based on the patient data and analyzed parameters."""
        ...

    def note(self, param: str, status: str) -> str:
        """Return a clinical note for a specific parameter and its status."""
        ...

    def rule_risk(self, data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic rule-based risk assessment."""
        ...

    def predict_risk(self, data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """ML or advanced risk prediction."""
        ...

    def score_symptoms(self, symptoms: List[str]) -> Dict[str, Any]:
        """Score symptoms against the disease profile."""
        ...
