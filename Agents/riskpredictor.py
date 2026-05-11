"""
Risk Predictor Agent — LangChain Implementation

Purpose:
    Checks patient-reported symptoms, assesses health risk level,
    and recommends the appropriate care pathway.

Steps:
    1. Collect symptoms, age, severity (1-10), duration, comorbidities via conversation
    2. Use symptom_lookup to retrieve matching conditions from medical knowledge base (FAISS + RAG)
    3. Use assess_risk to compute a risk score (rule-based engine)
    4. Use triage_route to map risk level to care pathway
    5. Append medical disclaimer to every final response

Exposes:
    RiskPredictorAgent                        -> class  (master agent / LangGraph interface)
        .predict_from_analysis(analysis)      -> str    (JSON string)
    run_risk_assessment(message, session_id)  -> dict   (conversational interface)
    get_risk_agent()                          -> AgentExecutor (raw LangChain agent)

Usage (master agent / LangGraph node):
    from Agents.riskpredictor import RiskPredictorAgent
    predictor = RiskPredictorAgent()
    risk_json = predictor.predict_from_analysis(analysis)

Usage (standalone conversational):
    from Agents.riskpredictor import run_risk_assessment
    result = run_risk_assessment("I have a headache and fever, age 45, severity 6")
"""

import os
import sys
import uuid
import re
import json
import logging
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
# Insert health_agent/ into sys.path so all internal imports
# (agent.tools, knowledge_base.retriever etc.) resolve correctly regardless
# of where the master agent calls this from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.join(_THIS_DIR, "health_agent")

if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

# ── Env setup ─────────────────────────────────────────────────────────────────
# Load .env from health_agent/ so GROQ_API_KEY, MEDICAL_KB_PATH etc.
# are available before any LangChain imports run.
from dotenv import load_dotenv
load_dotenv(os.path.join(_AGENT_DIR, ".env"))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Internal imports (only available after sys.path is set above) ─────────────
from agent.agent import build_agent  # noqa: E402


# ── Session store ─────────────────────────────────────────────────────────────
# Keeps one AgentExecutor per session so conversation memory persists
# across multiple turns. Replace with Redis in production.
_sessions: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_risk_level(text: str) -> Optional[str]:
    """
    Extract risk level keyword from free-text agent output.
    Returns one of: CRITICAL | HIGH | MEDIUM | LOW | None
    """
    patterns = [
        r'Level:\s*(CRITICAL|HIGH|MEDIUM|LOW)',
        r'risk level is\s*(CRITICAL|HIGH|MEDIUM|LOW)',
        r'risk level[:\s]+(CRITICAL|HIGH|MEDIUM|LOW)',
        r'assessed as\s*(CRITICAL|HIGH|MEDIUM|LOW)',
        r'risk is\s*(CRITICAL|HIGH|MEDIUM|LOW)',
        r'\b(CRITICAL|HIGH|MEDIUM|LOW)\s*risk\b',
        r'\b(CRITICAL|HIGH|MEDIUM|LOW)\b(?=\s*[-])',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _parse_risk_level_from_steps(steps: Optional[list]) -> Optional[str]:
    """
    Extract risk level from agent intermediate tool outputs (if available).
    """
    if not steps:
        return None
    for step in steps:
        if not isinstance(step, (list, tuple)) or len(step) != 2:
            continue
        action, observation = step
        tool_name = getattr(action, "tool", "")
        if tool_name == "assess_risk":
            level = _parse_risk_level(str(observation))
            if level:
                return level
    return None


def _extract_lab_values(analysis: dict) -> dict:
    """Extract core lab values from analyzer output or flat input."""
    values = {}
    params = analysis.get("parameters") or {}

    for key in ("hba1c", "glucose", "bmi", "age"):
        direct = analysis.get(key)
        if isinstance(direct, (int, float)):
            values[key] = direct

        param = params.get(key) or {}
        val = param.get("value")
        if isinstance(val, (int, float)):
            values[key] = val

    return values


def _compute_lab_risk(values: dict) -> dict:
    """Compute a lab-based risk score consistent with the master workflow."""
    hba1c = float(values.get("hba1c", 0) or 0)
    glucose = float(values.get("glucose", 0) or 0)
    bmi = float(values.get("bmi", 0) or 0)
    age = float(values.get("age", 0) or 0)

    hba1c_score = 0.0
    if hba1c >= 6.5:
        hba1c_score = 1.0
    elif hba1c >= 5.7:
        hba1c_score = 0.6

    glucose_score = 0.0
    if glucose >= 126:
        glucose_score = 1.0
    elif glucose >= 100:
        glucose_score = 0.6

    bmi_score = 0.0
    if bmi >= 40:
        bmi_score = 1.0
    elif bmi >= 35:
        bmi_score = 0.85
    elif bmi >= 30:
        bmi_score = 0.7
    elif bmi >= 25:
        bmi_score = 0.4

    age_score = 0.0
    if age >= 65:
        age_score = 0.8
    elif age >= 55:
        age_score = 0.6
    elif age >= 45:
        age_score = 0.4
    elif age >= 35:
        age_score = 0.2

    weighted_score = (
        (hba1c_score * 0.35)
        + (glucose_score * 0.35)
        + (bmi_score * 0.2)
        + (age_score * 0.1)
    )
    risk_score = int(round(weighted_score * 100))

    if risk_score <= 20:
        risk_level = "LOW"
    elif risk_score <= 40:
        risk_level = "MEDIUM"
    elif risk_score <= 60:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_probability": round(risk_score / 100.0, 2),
        "summary": f"Lab-based risk assessment: {risk_level} (Score: {risk_score}%).",
    }


def _level_to_probability(level: Optional[str]) -> float:
    """
    Map a risk level string to a probability float.
    Used by RiskPredictorAgent to satisfy the master agent's
    expected 'risk_probability' field in the output dict.
    """
    mapping = {
        "CRITICAL": 0.95,
        "HIGH":     0.75,
        "MEDIUM":   0.50,
        "LOW":      0.20,
    }
    return mapping.get(level, 0.0)


def _build_message_from_analysis(analysis: dict) -> str:
    """
    Convert a pre-analysed health data dict (from the report analyzer agent)
    into a natural language message the conversational risk agent can process.

    Handles whatever keys are present — nothing is assumed to be mandatory.
    """
    parts = []

    symptoms = analysis.get("symptoms") or analysis.get("abnormal_parameters") or []
    if isinstance(symptoms, list) and symptoms:
        parts.append("Symptoms / abnormal parameters: " + ", ".join(str(s) for s in symptoms))
    elif isinstance(symptoms, str) and symptoms:
        parts.append(f"Symptoms: {symptoms}")

    age = analysis.get("patient_age") or analysis.get("age")
    if age:
        parts.append(f"Age: {age}")

    severity = analysis.get("severity")
    if severity:
        parts.append(f"Severity: {severity}/10")

    duration = analysis.get("duration_days") or analysis.get("duration")
    if duration:
        parts.append(f"Duration: {duration} days")

    comorbidities = analysis.get("comorbidities") or analysis.get("conditions") or []
    if isinstance(comorbidities, list) and comorbidities:
        parts.append("Comorbidities: " + ", ".join(str(c) for c in comorbidities))
    elif isinstance(comorbidities, str) and comorbidities:
        parts.append(f"Comorbidities: {comorbidities}")

    lab_values = _extract_lab_values(analysis)
    lab_parts = []
    if "hba1c" in lab_values:
        lab_parts.append(f"HbA1c {lab_values['hba1c']}%")
    if "glucose" in lab_values:
        lab_parts.append(f"Glucose {lab_values['glucose']} mg/dL")
    if "bmi" in lab_values:
        lab_parts.append(f"BMI {lab_values['bmi']}")
    if "age" in lab_values:
        lab_parts.append(f"Age {lab_values['age']}")
    if lab_parts:
        parts.append("Lab values: " + ", ".join(lab_parts))

    # Fallback — if nothing recognisable, pass the raw dict as context
    if not parts:
        parts.append(f"Patient analysis data: {json.dumps(analysis)}")

    return ". ".join(parts) + "."


# ── Compatibility shim ────────────────────────────────────────────────────────

class RiskPredictorAgent:
    """
    Compatibility shim — satisfies the master agent's expected interface:

        from Agents.riskpredictor import RiskPredictorAgent
        predictor = RiskPredictorAgent()
        risk_json = predictor.predict_from_analysis(analysis)

    This class does NOT rewrite any project logic.
    It translates the master agent's dict input into a natural language
    message and delegates everything to run_risk_assessment().

    Your project tree (agent.py, tools.py, memory.py, knowledge_base/)
    remains completely untouched.
    """

    def predict_from_analysis(self, analysis: dict) -> str:
        """
        Args:
            analysis : dict from the report analyzer agent containing
                       patient health data (symptoms, age, severity etc.)

        Returns:
            JSON string with keys:
                risk_level        : CRITICAL | HIGH | MEDIUM | LOW | None
                risk_probability  : float (0.0 - 1.0)
                response          : full agent response text with disclaimer
                success           : bool
                error             : str | None
        """
        message = _build_message_from_analysis(analysis)
        logger.info(f"[RiskPredictorAgent] Built message from analysis: {message}")

        result = run_risk_assessment(message)
        lab_values = _extract_lab_values(analysis)
        lab_risk = None

        if result["risk_level"] is None and lab_values:
            lab_risk = _compute_lab_risk(lab_values)
            result["risk_level"] = lab_risk["risk_level"]
            result["success"] = True
            result["error"] = None

            if result.get("response"):
                result["response"] = f"{result['response']}\n\n{lab_risk['summary']}"
            else:
                result["response"] = lab_risk["summary"]

        output = {
            "risk_level":       result["risk_level"],
            "risk_probability": lab_risk["risk_probability"] if lab_risk else _level_to_probability(result["risk_level"]),
            "response":         result["response"],
            "success":          result["success"],
            "error":            result["error"],
        }

        return json.dumps(output)


# ── Public API ────────────────────────────────────────────────────────────────

def run_risk_assessment(message: str, session_id: Optional[str] = None) -> dict:
    """
    Conversational entry point. Accepts a natural language message,
    runs it through the LangChain agent, and returns a structured dict.

    Args:
        message    : User's natural language input
        session_id : Optional. Pass back the session_id from the previous
                     turn to continue the conversation with memory intact.

    Returns:
        {
            "session_id" : str,
            "response"   : str,           # full response with disclaimer
            "risk_level" : str | None,    # CRITICAL | HIGH | MEDIUM | LOW
            "success"    : bool,
            "error"      : str | None
        }
    """
    if not session_id or str(session_id).lower() == 'null':
        session_id = str(uuid.uuid4())

    if session_id not in _sessions:
        logger.info(f"[RiskPredictor] New session: {session_id}")
        _sessions[session_id] = build_agent()

    agent = _sessions[session_id]

    try:
        result = agent.invoke({"input": message})
        output = result.get("output", "")
        risk_level = _parse_risk_level(output)

        if not risk_level:
            risk_level = _parse_risk_level_from_steps(result.get("intermediate_steps"))

        success = risk_level is not None
        error = None if success else "Risk level not found in agent output"

        logger.info(f"[RiskPredictor] Session {session_id} | Risk: {risk_level}")

        return {
            "session_id": session_id,
            "response":   output,
            "risk_level": risk_level,
            "success":    success,
            "error":      error,
        }

    except Exception as e:
        logger.error(f"[RiskPredictor] Error in session {session_id}: {e}")
        return {
            "session_id": session_id,
            "response":   None,
            "risk_level": None,
            "success":    False,
            "error":      str(e),
        }


def get_risk_agent(session_id: Optional[str] = None):
    """
    Returns the raw AgentExecutor for direct LangGraph node integration.
    Use when the master agent wants to manage invoke() itself.
    """
    if session_id and session_id in _sessions:
        return _sessions[session_id]

    agent = build_agent()

    if session_id:
        _sessions[session_id] = agent

    return agent


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Risk Predictor Agent — Standalone Test ===\n")

    # Test 1: Conversational interface
    print("-- Test 1: run_risk_assessment (conversational) --")
    result = run_risk_assessment("I have a severe headache and dizziness")
    session = result["session_id"]
    print(f"Agent: {result['response']}\n")

    result2 = run_risk_assessment(
        message="Severity is 7, I am 58 years old, have hypertension, started 2 days ago",
        session_id=session
    )
    print(f"Agent: {result2['response']}")
    print(f"Risk Level: {result2['risk_level']}\n")

    # Test 2: Master agent compatibility interface
    print("-- Test 2: RiskPredictorAgent.predict_from_analysis (master agent shim) --")
    predictor = RiskPredictorAgent()
    mock_analysis = {
        "symptoms":      ["headache", "dizziness"],
        "patient_age":   58,
        "severity":      7,
        "duration_days": 2,
        "comorbidities": ["hypertension"],
    }
    risk_json = predictor.predict_from_analysis(mock_analysis)
    risk = json.loads(risk_json)
    print(f"Risk Level:       {risk['risk_level']}")
    print(f"Risk Probability: {risk['risk_probability']}")
    print(f"Success:          {risk['success']}")