import json
from langchain_core.tools import tool
from knowledge_base.retriever import get_retriever


@tool
def assess_risk(patient_data: str) -> str:
    """Assess patient risk level from structured data.
    Input: JSON string with keys: symptoms (list), age (int), comorbidities (list),
           duration_days (int), severity (1-10).
    Output: risk level string with reasoning.
    """
    data = json.loads(patient_data)
    score = 0

    # Rule-based scoring
    if data.get('severity', 0) >= 8:
        score += 40
    elif data.get('severity', 0) >= 6:
        score += 20

    if data.get('age', 0) > 65:
        score += 20

    if data.get('duration_days', 0) > 7:
        score += 15

    high_risk_comorbidities = ['diabetes', 'hypertension', 'heart_disease', 'immunocompromised']
    for c in data.get('comorbidities', []):
        if c.lower() in high_risk_comorbidities:
            score += 15

    # Determine level
    if score >= 70:
        level = 'CRITICAL - Seek Emergency Care Immediately'
    elif score >= 45:
        level = 'HIGH - Consult a Doctor Within 24 Hours'
    elif score >= 20:
        level = 'MEDIUM - Schedule a Doctor Appointment'
    else:
        level = 'LOW - Monitor Symptoms, Self-Care Appropriate'

    return f'Risk Score: {score}/100 | Level: {level}'


@tool
def triage_route(risk_level: str) -> str:
    """Recommend appropriate care pathway based on risk level.
    Input: risk level string (LOW / MEDIUM / HIGH / CRITICAL).
    Output: actionable triage recommendation.
    """
    risk_level = risk_level.upper()
    if 'CRITICAL' in risk_level:
        return 'EMERGENCY: Call 911 or go to nearest Emergency Room immediately.'
    elif 'HIGH' in risk_level:
        return 'URGENT CARE: Visit urgent care or telemedicine within 24 hours.'
    elif 'MEDIUM' in risk_level:
        return 'PRIMARY CARE: Schedule appointment with your GP within 2-3 days.'
    else:
        return 'SELF-CARE: Rest, hydrate, and monitor. Return if symptoms worsen.'


@tool
def add_disclaimer(response: str) -> str:
    """Append a required medical disclaimer to any health response."""
    disclaimer = (
        "\n\n--- IMPORTANT MEDICAL DISCLAIMER ---\n"
        "This information is for educational purposes only and does not constitute "
        "medical advice. Always consult a qualified healthcare professional for "
        "diagnosis and treatment. In an emergency, call 911 immediately."
    )
    return response + disclaimer