"""
Symptom Analyzer Agent — Diabetes
==================================
Responsibilities:
  1. map_symptoms_to_conditions  — LangChain @tool
  2. Clinical Reasoning          — LangChain pipe chain (DeepSeek-R1:8b)

Covers all diabetes types and diabetes-related conditions:
  - Type 1 Diabetes
  - Type 2 Diabetes
  - Pre-diabetes / Insulin Resistance
  - Gestational Diabetes
  - Hypoglycemia
  - Diabetic Peripheral Neuropathy
  - Diabetic Autonomic Neuropathy
  - Diabetic Ketoacidosis (DKA)

Does NOT validate lab values or classify risk scores.
Those belong to ReportAnalyzerAgent and RiskPredictorAgent.

Inputs (any combination):
  - symptoms       : list of symptom strings (checkboxes or parsed text)
  - report_context : dict from ReportAnalyzerAgent
  - risk_context   : dict from RiskPredictorAgent
  - manual_text    : free-text from user
"""

import re
import logging
import requests
from typing import List, Dict, Any, Optional

from langchain_community.llms import Ollama
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

OLLAMA_TAGS = "http://localhost:11434/api/tags"

# DeepSeek-R1 uses Chain-of-Thought — <think> blocks are stripped by _clean_llm_output()
REASONING_MODEL = {
    "id":       "deepseek-r1:8b",
    "name":     "DeepSeek-R1 8B",
    "pull_cmd": "ollama pull deepseek-r1:8b",
}


# ── Symptom Taxonomy — Diabetes Only (56 symptoms, 6 categories) ─────────────

SYMPTOM_CATEGORIES: Dict[str, List[str]] = {
    "Hyperglycemia / High Blood Sugar": [
        "Polydipsia (excessive thirst)",
        "Polyuria (frequent urination)",
        "Polyphagia (excessive hunger)",
        "Unexplained weight loss",
        "Fatigue / Low energy",
        "Blurred or fluctuating vision",
        "Slow-healing wounds or cuts",
        "Recurrent infections (skin, UTI, yeast)",
        "Fruity / acetone breath",
        "Nausea or vomiting",
        "Dry mouth",
        "Headaches",
    ],
    "Hypoglycemia / Low Blood Sugar": [
        "Shakiness / trembling",
        "Sweating without exertion",
        "Palpitations / rapid heartbeat",
        "Sudden anxiety or irritability",
        "Confusion / difficulty concentrating",
        "Dizziness or lightheadedness",
        "Hunger after recent meal",
        "Pallor (pale skin)",
        "Weakness / feeling faint",
        "Nightmares or night sweats",
    ],
    "Insulin Resistance / Pre-diabetes": [
        "Central obesity / abdominal fat",
        "Acanthosis nigricans (dark skin patches on neck/armpits)",
        "Fatigue after carbohydrate meals",
        "Brain fog / poor concentration",
        "Cravings for sweets / carbohydrates",
        "Elevated blood pressure",
        "Fatty liver symptoms (right upper abdominal discomfort)",
        "Skin tags",
    ],
    "Type 1 Diabetes / DKA": [
        "Rapid onset of thirst and urination",
        "Significant unintentional weight loss",
        "Fruity / acetone breath",
        "Deep laboured breathing (Kussmaul breathing)",
        "Severe nausea or vomiting",
        "Abdominal pain",
        "Extreme fatigue or lethargy",
        "Confusion or altered consciousness",
    ],
    "Diabetic Peripheral Neuropathy": [
        "Tingling or numbness in feet or hands",
        "Burning pain in lower extremities",
        "Decreased sensation in feet",
        "Sharp or electric shock-like pains in legs",
        "Increased sensitivity to touch",
        "Foot ulcers or sores not healing",
        "Muscle weakness in feet or legs",
    ],
    "Diabetic Autonomic Neuropathy": [
        "Postural hypotension (dizziness on standing)",
        "Gastroparesis symptoms (early satiety, bloating, nausea after eating)",
        "Erectile dysfunction",
        "Bladder dysfunction (incomplete emptying, incontinence)",
        "Excessive or reduced sweating",
        "Resting tachycardia (fast heart rate at rest)",
        "Hypoglycemia unawareness (no warning symptoms before low blood sugar)",
    ],
}

# Maps diabetes conditions to their known symptom profiles
_CONDITION_SYMPTOM_MAP: Dict[str, List[str]] = {
    "Type 2 Diabetes": [
        "Polydipsia (excessive thirst)",
        "Polyuria (frequent urination)",
        "Polyphagia (excessive hunger)",
        "Unexplained weight loss",
        "Fatigue / Low energy",
        "Blurred or fluctuating vision",
        "Slow-healing wounds or cuts",
        "Recurrent infections (skin, UTI, yeast)",
        "Dry mouth",
        "Headaches",
    ],
    "Type 1 Diabetes": [
        "Rapid onset of thirst and urination",
        "Significant unintentional weight loss",
        "Fruity / acetone breath",
        "Extreme fatigue or lethargy",
        "Blurred or fluctuating vision",
        "Recurrent infections (skin, UTI, yeast)",
        "Nausea or vomiting",
        "Abdominal pain",
    ],
    "Pre-diabetes / Insulin Resistance": [
        "Fatigue after carbohydrate meals",
        "Brain fog / poor concentration",
        "Central obesity / abdominal fat",
        "Acanthosis nigricans (dark skin patches on neck/armpits)",
        "Cravings for sweets / carbohydrates",
        "Elevated blood pressure",
        "Skin tags",
    ],
    "Gestational Diabetes": [
        "Polydipsia (excessive thirst)",
        "Polyuria (frequent urination)",
        "Fatigue / Low energy",
        "Blurred or fluctuating vision",
        "Recurrent infections (skin, UTI, yeast)",
        "Nausea or vomiting",
    ],
    "Hypoglycemia": [
        "Shakiness / trembling",
        "Sweating without exertion",
        "Palpitations / rapid heartbeat",
        "Sudden anxiety or irritability",
        "Hunger after recent meal",
        "Dizziness or lightheadedness",
        "Confusion / difficulty concentrating",
        "Pallor (pale skin)",
        "Weakness / feeling faint",
        "Nightmares or night sweats",
    ],
    "Diabetic Ketoacidosis (DKA)": [
        "Fruity / acetone breath",
        "Deep laboured breathing (Kussmaul breathing)",
        "Severe nausea or vomiting",
        "Abdominal pain",
        "Extreme fatigue or lethargy",
        "Confusion or altered consciousness",
        "Rapid onset of thirst and urination",
        "Significant unintentional weight loss",
    ],
    "Diabetic Peripheral Neuropathy": [
        "Tingling or numbness in feet or hands",
        "Burning pain in lower extremities",
        "Decreased sensation in feet",
        "Sharp or electric shock-like pains in legs",
        "Increased sensitivity to touch",
        "Foot ulcers or sores not healing",
        "Muscle weakness in feet or legs",
    ],
    "Diabetic Autonomic Neuropathy": [
        "Postural hypotension (dizziness on standing)",
        "Gastroparesis symptoms (early satiety, bloating, nausea after eating)",
        "Erectile dysfunction",
        "Bladder dysfunction (incomplete emptying, incontinence)",
        "Excessive or reduced sweating",
        "Resting tachycardia (fast heart rate at rest)",
        "Hypoglycemia unawareness (no warning symptoms before low blood sugar)",
    ],
}


# ── LangChain Tool ────────────────────────────────────────────────────────────

@tool
def map_symptoms_to_conditions(symptoms: List[str]) -> Dict[str, Any]:
    """
    Map patient symptoms to ranked diabetes condition hypotheses.

    Returns condition_hypotheses (sorted by match_count), top_hypothesis,
    total_symptoms_checked, and unmatched_symptoms.
    """
    if not symptoms:
        return {
            "condition_hypotheses":   [],
            "top_hypothesis":         "No symptoms provided",
            "total_symptoms_checked": 0,
            "unmatched_symptoms":     [],
        }

    scored: Dict[str, Dict] = {}
    all_mapped: set = set()

    for condition, cond_syms in _CONDITION_SYMPTOM_MAP.items():
        matched = [s for s in symptoms if s in cond_syms]
        if matched:
            scored[condition] = {
                "matched_symptoms": matched,
                "match_count":      len(matched),
                "coverage":         f"{len(matched)}/{len(cond_syms)}",
            }
            all_mapped.update(matched)

    ranked    = sorted(scored.items(), key=lambda x: x[1]["match_count"], reverse=True)
    unmatched = [s for s in symptoms if s not in all_mapped]

    return {
        "condition_hypotheses":   [{"condition": k, **v} for k, v in ranked],
        "top_hypothesis":         ranked[0][0] if ranked else "No condition mapped",
        "total_symptoms_checked": len(symptoms),
        "unmatched_symptoms":     unmatched,
    }


# ── Context Builder ───────────────────────────────────────────────────────────

def build_context(
    report_context: Optional[Dict] = None,
    risk_context:   Optional[Dict] = None,
    manual_text:    Optional[str]  = None,
) -> str:
    """
    Normalise upstream agent outputs and free text into a single prompt string.
    Accepts any combination of the three inputs.
    """
    parts: List[str] = []

    if report_context:
        params = report_context.get("parameters", {})
        if params:
            parts.append("=== LAB REPORT (from ReportAnalyzerAgent) ===")
            for k, v in params.items():
                if isinstance(v, dict):
                    line = f"  {k.upper()}: {v.get('value','')} {v.get('unit','')} [{v.get('status','')}]"
                    if v.get("note"):
                        line += f" — {v['note']}"
                    parts.append(line)
                else:
                    parts.append(f"  {k.upper()}: {v}")
        flags = report_context.get("endocrine_flags", [])
        if flags:
            parts.append(f"  Flags: {', '.join(flags)}")
        abnormal = report_context.get("abnormal_parameters", [])
        if abnormal:
            parts.append(f"  Abnormal Parameters: {', '.join(p.upper() for p in abnormal)}")

    if risk_context:
        parts.append("=== RISK PREDICTION (from RiskPredictorAgent) ===")
        if risk_context.get("risk_tier"):
            parts.append(f"  Risk Tier:  {risk_context['risk_tier']}")
        if risk_context.get("risk_score"):
            parts.append(f"  Risk Score: {risk_context['risk_score']}/100")
        if risk_context.get("ada_classifications"):
            parts.append(f"  ADA Classification: {'; '.join(risk_context['ada_classifications'])}")
        if risk_context.get("recommended_action"):
            parts.append(f"  Recommended Action: {risk_context['recommended_action']}")
        # Pass through any extra fields from RiskPredictorAgent
        for k, v in risk_context.items():
            if k not in ("risk_tier", "risk_score", "ada_classifications", "recommended_action"):
                parts.append(f"  {k}: {v}")

    if manual_text and manual_text.strip():
        parts.append("=== ADDITIONAL PATIENT NOTES (manual input) ===")
        parts.append(f"  {manual_text.strip()}")

    return "\n".join(parts) if parts else "No upstream context provided."


# ── LangChain Components ──────────────────────────────────────────────────────

def _clean_llm_output(text: str) -> str:
    """
    Post-processes LLM output:
      - Strips DeepSeek-R1 <think>...</think> CoT blocks
      - Strips markdown headers (# ## ###) that models sometimes add
    """
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,3}\s+", "", text, flags=re.MULTILINE)
    return text.strip()


SYMPTOM_REASONING_PROMPT = PromptTemplate(
    input_variables=[
        "upstream_context",
        "symptom_map",
        "symptoms_text",
        "top_hypothesis",
        "unmatched_symptoms",
    ],
    template="""You are a specialist clinical decision-support system for Diabetes.
Your role is ONLY diabetes symptom analysis and clinical reasoning.
Lab validation and risk scoring have already been done by upstream agents — use their output as context.

=== UPSTREAM AGENT CONTEXT ===
{upstream_context}

=== PATIENT SYMPTOMS ===
{symptoms_text}

=== SYMPTOM → DIABETES CONDITION MAPPING ===
Top hypothesis: {top_hypothesis}
{symptom_map}

Unmatched symptoms (not in any known diabetes pattern): {unmatched_symptoms}

=== YOUR TASK ===
Write a focused diabetes symptom-analysis note. Use EXACTLY this structure, no more than 200 words total:

1. SYMPTOM INTERPRETATION: Which symptoms are most clinically significant for diabetes and why.
2. CONDITION CORRELATION: How the symptoms align with the upstream lab/risk findings.
3. PRIMARY HYPOTHESIS: Most likely diabetes type or condition based on the full picture.
4. DIFFERENTIAL: 1–2 alternative diabetes conditions the unmatched or overlapping symptoms may suggest.
5. CLINICAL RECOMMENDATION: Specific next steps — which specialist, which tests, which lifestyle or medication changes.

Be concise and clinically precise. Do not repeat lab values verbatim -- refer to them by finding (e.g. "elevated HbA1c").
Do not add AI disclaimers or suggest consulting a doctor generically -- give specific specialty referrals.
Do not use markdown formatting, headers, or bullet symbols. Write in plain numbered sections only.""",
)


def _build_chain():
    """Check Ollama for DeepSeek-R1 and return a ready chain, or None if unavailable."""
    try:
        resp      = requests.get(OLLAMA_TAGS, timeout=4)
        available = [m["name"] for m in resp.json().get("models", [])]
        model_id  = next(
            (a for a in available if REASONING_MODEL["id"].split(":")[0] in a),
            None,
        )
        if not model_id:
            logger.warning(f"DeepSeek-R1 not found. Run: {REASONING_MODEL['pull_cmd']}")
            return None

        # Modern pipe syntax — replaces deprecated LLMChain
        llm   = Ollama(model=model_id, temperature=0.2, num_predict=1200)
        chain = SYMPTOM_REASONING_PROMPT | llm | StrOutputParser()
        logger.info(f"SymptomAnalyzerAgent using: {model_id}")
        return chain

    except Exception as e:
        logger.warning(f"Ollama unavailable: {e}")
        return None


# ── Rule-Based Fallback ───────────────────────────────────────────────────────

def _rule_based_reasoning(
    symptom_map_result: Dict,
    upstream_context:   str,
    symptoms:           List[str],
) -> str:
    """Structured diabetes clinical note without LLM. Used when Ollama is unavailable."""

    hyps      = symptom_map_result.get("condition_hypotheses", [])
    top       = symptom_map_result.get("top_hypothesis", "")
    unmatched = symptom_map_result.get("unmatched_symptoms", [])
    lines     = []

    if not symptoms:
        lines.append("ℹ️ No symptoms were provided.")
        if upstream_context and "No upstream context" not in upstream_context:
            lines.append("\nReasoning is based on upstream agent context only:")
            lines.append(upstream_context)
        lines.append("\nRecommendation: Select symptoms from the checklist or enter a text description.")
        return "\n".join(lines)

    lines.append(f"🩺 SYMPTOM INTERPRETATION ({len(symptoms)} reported)")
    for s in symptoms:
        lines.append(f"  - {s}")

    if hyps:
        lines.append(f"\n🔍 PRIMARY HYPOTHESIS: {top}")
        lines.append(f"  Matched: {', '.join(hyps[0]['matched_symptoms'])}")
        lines.append(f"  Coverage: {hyps[0]['coverage']} known symptoms for this condition")
        if len(hyps) > 1:
            lines.append("\n🔀 DIFFERENTIAL:")
            for h in hyps[1:3]:
                lines.append(f"  • {h['condition']} — {h['match_count']} symptom(s) matched")
    else:
        lines.append("\n⚠️ No symptoms matched any known diabetes condition pattern.")
        lines.append("Consider broader diabetes workup or specialist evaluation.")

    if unmatched:
        lines.append(f"\n❓ UNMATCHED SYMPTOMS: {', '.join(unmatched)}")
        lines.append("  These may indicate non-diabetes causes — consider broader evaluation.")

    if upstream_context and "No upstream context" not in upstream_context:
        lines.append("\n📋 UPSTREAM CONTEXT SUMMARY:")
        for line in upstream_context.split("\n"):
            if any(kw in line for kw in ["Flag", "Risk Tier", "ADA", "Abnormal"]):
                lines.append(f"  {line.strip()}")

    recs = {
        "Type 2 Diabetes":               "Refer to Endocrinologist. Confirm with repeat HbA1c + OGTT. Initiate MNT and consider Metformin.",
        "Type 1 Diabetes":               "Urgent referral to Endocrinologist. Check C-peptide, GAD antibodies, fasting insulin. Initiate insulin therapy.",
        "Pre-diabetes / Insulin Resistance": "Lifestyle intervention program. Repeat HbA1c in 3 months. Screen for metabolic syndrome.",
        "Gestational Diabetes":          "Refer to OB-GYN + Endocrinologist. OGTT immediately. Monitor fetal growth. Dietary modification first line.",
        "Hypoglycemia":                  "Evaluate for reactive vs. fasting hypoglycemia. 72-hour fast test if indicated. Review current medications.",
        "Diabetic Ketoacidosis (DKA)":   "EMERGENCY — immediate hospital admission. IV fluids, insulin drip, electrolyte correction. Monitor hourly.",
        "Diabetic Peripheral Neuropathy": "Refer to Neurologist. Monofilament foot exam. Optimize glycemic control. Consider gabapentin or duloxetine.",
        "Diabetic Autonomic Neuropathy": "Refer to Neurologist + Gastroenterologist. Orthostatic BP measurement. HRV testing. Optimize glycemic control.",
    }
    lines.append(f"\n✅ RECOMMENDATION: {recs.get(top, 'Refer to Endocrinologist for comprehensive diabetes evaluation.')}")

    return "\n".join(lines)


# ── Main Entry Point ──────────────────────────────────────────────────────────

def analyze_symptoms(
    symptoms:       List[str],
    report_context: Optional[Dict] = None,
    risk_context:   Optional[Dict] = None,
    manual_text:    Optional[str]  = None,
    use_llm:        bool           = True,
) -> Dict[str, Any]:
    """
    Run the full diabetes symptom analysis pipeline.

    Args:
        symptoms:       Symptom strings from checkboxes or text parsing.
        report_context: Output dict from ReportAnalyzerAgent (optional).
        risk_context:   Output dict from RiskPredictorAgent (optional).
        manual_text:    Free-text notes from user (optional).
        use_llm:        Set False to force rule-based mode.

    Returns:
        {
          "symptom_mapping": { condition_hypotheses, top_hypothesis, ... },
          "reasoning":       str,
          "model_used":      "deepseek-r1:8b" | "rule-based",
          "input_summary":   { symptoms_count, sources_used, top_hypothesis }
        }
    """
    symptom_map_result = map_symptoms_to_conditions.invoke({"symptoms": symptoms})
    upstream_context   = build_context(report_context, risk_context, manual_text)

    top_hyp        = symptom_map_result.get("top_hypothesis", "No condition mapped")
    hyps           = symptom_map_result.get("condition_hypotheses", [])
    unmatched      = symptom_map_result.get("unmatched_symptoms", [])
    symp_text      = "\n".join(f"  - {s}" for s in symptoms) if symptoms else "  None provided."
    hyp_text       = "\n".join(
        f"  • {h['condition']}: {h['match_count']} symptom(s) matched "
        f"({', '.join(h['matched_symptoms'][:3])}{'...' if len(h['matched_symptoms']) > 3 else ''})"
        for h in hyps[:5]
    ) or "  No condition matches found."
    unmatched_text = ", ".join(unmatched) if unmatched else "None"

    reasoning  = ""
    model_used = "rule-based"
    fallback   = _rule_based_reasoning(symptom_map_result, upstream_context, symptoms)

    if use_llm:
        chain = _build_chain()
        if chain:
            try:
                llm_output = chain.invoke({
                    "upstream_context":   upstream_context,
                    "symptom_map":        hyp_text,
                    "symptoms_text":      symp_text,
                    "top_hypothesis":     top_hyp,
                    "unmatched_symptoms": unmatched_text,
                })
                # Apply cleaning (strips <think> blocks and markdown headers)
                llm_output = _clean_llm_output(llm_output if isinstance(llm_output, str) else "")
                if llm_output and len(llm_output.strip()) > 80:
                    reasoning  = llm_output.strip()
                    model_used = REASONING_MODEL["id"]
                    logger.info(f"Reasoning generated via {REASONING_MODEL['name']}")
                else:
                    logger.warning("LLM output too short -- using rule-based fallback")
            except Exception as e:
                logger.warning(f"LLM chain failed: {e} -- using rule-based fallback")

    # Always fall back if LLM produced nothing
    if not reasoning or not reasoning.strip():
        reasoning = fallback

    sources_used = []
    if report_context: sources_used.append("ReportAnalyzerAgent")
    if risk_context:   sources_used.append("RiskPredictorAgent")
    if manual_text:    sources_used.append("ManualText")
    if symptoms:       sources_used.append("SymptomCheckboxes")

    return {
        "symptom_mapping": symptom_map_result,
        "reasoning":       reasoning,
        "model_used":      model_used,
        "input_summary": {
            "symptoms_count": len(symptoms),
            "sources_used":   sources_used,
            "top_hypothesis": top_hyp,
        },
    }


__all__ = [
    "analyze_symptoms",
    "map_symptoms_to_conditions",
    "build_context",
    "SYMPTOM_CATEGORIES",
    "REASONING_MODEL",
]


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    mock_report = {
        "parameters": {
            "hba1c":   {"value": 6.8,  "unit": "%",     "status": "High", "note": "Diabetic range (ADA)"},
            "glucose": {"value": 148,  "unit": "mg/dL", "status": "High", "note": "FPG diabetes criterion"},
            "bmi":     {"value": 29.1, "unit": "kg/m²", "status": "High", "note": "Overweight"},
            "homa_ir": {"value": 5.2,  "unit": "index", "status": "High", "note": "Severe insulin resistance"},
        },
        "abnormal_parameters": ["hba1c", "glucose", "bmi", "homa_ir"],
        "endocrine_flags": ["DIABETES CRITERIA MET", "SEVERE INSULIN RESISTANCE"],
    }
    mock_risk = {
        "risk_tier":           "HIGH",
        "risk_score":          82,
        "ada_classifications": ["Diabetes Mellitus (HbA1c ≥ 6.5%)", "Diabetes Mellitus (FPG ≥ 126 mg/dL)"],
        "recommended_action":  "Urgent endocrinology referral. Initiate diabetes management protocol.",
    }
    symptoms = [
        "Polydipsia (excessive thirst)",
        "Polyuria (frequent urination)",
        "Fatigue after carbohydrate meals",
        "Blurred or fluctuating vision",
        "Central obesity / abdominal fat",
        "Brain fog / poor concentration",
        "Tingling or numbness in feet or hands",
    ]

    result = analyze_symptoms(
        symptoms       = symptoms,
        report_context = mock_report,
        risk_context   = mock_risk,
        manual_text    = "Symptoms worsening over 3 months. Family history of Type 2 diabetes.",
        use_llm        = True,
    )

    print("\n=== SYMPTOM MAPPING ===")
    for h in result["symptom_mapping"]["condition_hypotheses"]:
        print(f"  {h['condition']}: {h['match_count']} matched ({h['coverage']})")
    print(f"\n=== MODEL USED: {result['model_used']} ===")
    print(f"\n=== CLINICAL REASONING ===\n{result['reasoning']}")
    print(f"\n=== INPUT SUMMARY ===\n{json.dumps(result['input_summary'], indent=2)}")
