from typing import List, Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod

class DiseasePlugin(ABC):
    """
    Base class for all disease-specific plugins.
    Each plugin encapsulates the clinical logic for a specific condition.
    """

    name: str
    required_fields: List[str]
    optional_fields: List[str]
    reference_ranges: Dict[str, Dict[str, Any]]
    trend_scales: Dict[str, float]
    symptom_profiles: Dict[str, Any]
    alert_thresholds: Dict[str, float]

    @abstractmethod
    def flags(self, data: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
        """Analyzes parameters against ranges to return clinical flags."""
        pass

    @abstractmethod
    def notes(self, param: str, status: str) -> str:
        """Returns clinical notes based on parameter status (Normal/High/Low)."""
        pass

    @abstractmethod
    def rule_risk(self, data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic scoring based on lab values."""
        pass

    @abstractmethod
    def predict_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """ML-based risk prediction, falls back to rule_risk."""
        pass

    @abstractmethod
    def score_symptoms(self, symptoms: List[str]) -> Dict[str, Any]:
        """Matches user symptoms against profiles to return a severity score."""
        pass
