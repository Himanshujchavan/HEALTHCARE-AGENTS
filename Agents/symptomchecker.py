"""
Symptom Analyzer Agent — Diabetes (v3)
=======================================
Upgrades over v2:
  1. Semantic symptom matching  — TF-IDF + medical synonym expansion
                                   replaces exact string equality
  2. Dataset integration        — Pima Indians Diabetes Dataset (UCI, 768 patients)
                                   trains a LogisticRegression classifier that
                                   predicts diabetes probability from lab values
  3. Accuracy measurement       — precision, recall, F1, ROC-AUC, confusion matrix
                                   computed via 5-fold cross-validation on Pima data
  4. Clinical reasoning         — DeepSeek-R1 8B (unchanged, Ollama)

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

Dataset reference:
  Smith, J.W. et al. (1988). Using the ADAP learning algorithm to forecast the
  onset of diabetes mellitus. Proceedings of the Symposium on Computer Applications
  in Medical Care (pp. 261-265). UCI ML Repository — public domain.

Inputs (any combination):
  - symptoms       : list of symptom strings (checkboxes OR free-text paraphrases)
  - lab_values     : dict for Pima model {"glucose":148,"bmi":29,"age":45,...}
  - report_context : dict from ReportAnalyzerAgent
  - risk_context   : dict from RiskPredictorAgent
  - manual_text    : free-text from user
"""

import re
import logging
import pickle
import warnings
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler

from langchain_community.llms import Ollama
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os

_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "pima_diabetes.csv"
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

OLLAMA_TAGS = "http://localhost:11434/api/tags"

REASONING_MODEL = {
    "id":       "deepseek-r1:8b",
    "name":     "DeepSeek-R1 8B",
    "pull_cmd": "ollama pull deepseek-r1:8b",
}

# Path where the trained Pima model is cached after first training
_MODEL_CACHE_PATH = Path(__file__).parent / "pima_model_cache.pkl"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — SYMPTOM TAXONOMY (unchanged from v2)
# ══════════════════════════════════════════════════════════════════════════════

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

_CONDITION_SYMPTOM_MAP: Dict[str, List[str]] = {
    "Type 2 Diabetes": [
        "Polydipsia (excessive thirst)", "Polyuria (frequent urination)",
        "Polyphagia (excessive hunger)", "Unexplained weight loss",
        "Fatigue / Low energy", "Blurred or fluctuating vision",
        "Slow-healing wounds or cuts", "Recurrent infections (skin, UTI, yeast)",
        "Dry mouth", "Headaches",
    ],
    "Type 1 Diabetes": [
        "Rapid onset of thirst and urination", "Significant unintentional weight loss",
        "Fruity / acetone breath", "Extreme fatigue or lethargy",
        "Blurred or fluctuating vision", "Recurrent infections (skin, UTI, yeast)",
        "Nausea or vomiting", "Abdominal pain",
    ],
    "Pre-diabetes / Insulin Resistance": [
        "Fatigue after carbohydrate meals", "Brain fog / poor concentration",
        "Central obesity / abdominal fat",
        "Acanthosis nigricans (dark skin patches on neck/armpits)",
        "Cravings for sweets / carbohydrates", "Elevated blood pressure", "Skin tags",
    ],
    "Gestational Diabetes": [
        "Polydipsia (excessive thirst)", "Polyuria (frequent urination)",
        "Fatigue / Low energy", "Blurred or fluctuating vision",
        "Recurrent infections (skin, UTI, yeast)", "Nausea or vomiting",
    ],
    "Hypoglycemia": [
        "Shakiness / trembling", "Sweating without exertion",
        "Palpitations / rapid heartbeat", "Sudden anxiety or irritability",
        "Hunger after recent meal", "Dizziness or lightheadedness",
        "Confusion / difficulty concentrating", "Pallor (pale skin)",
        "Weakness / feeling faint", "Nightmares or night sweats",
    ],
    "Diabetic Ketoacidosis (DKA)": [
        "Fruity / acetone breath", "Deep laboured breathing (Kussmaul breathing)",
        "Severe nausea or vomiting", "Abdominal pain",
        "Extreme fatigue or lethargy", "Confusion or altered consciousness",
        "Rapid onset of thirst and urination", "Significant unintentional weight loss",
    ],
    "Diabetic Peripheral Neuropathy": [
        "Tingling or numbness in feet or hands", "Burning pain in lower extremities",
        "Decreased sensation in feet", "Sharp or electric shock-like pains in legs",
        "Increased sensitivity to touch", "Foot ulcers or sores not healing",
        "Muscle weakness in feet or legs",
    ],
    "Diabetic Autonomic Neuropathy": [
        "Postural hypotension (dizziness on standing)",
        "Gastroparesis symptoms (early satiety, bloating, nausea after eating)",
        "Erectile dysfunction", "Bladder dysfunction (incomplete emptying, incontinence)",
        "Excessive or reduced sweating", "Resting tachycardia (fast heart rate at rest)",
        "Hypoglycemia unawareness (no warning symptoms before low blood sugar)",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — SEMANTIC SYMPTOM MATCHING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# Medical synonym dictionary — maps patient lay language to canonical medical terms.
# This is what makes "always thirsty" match "Polydipsia (excessive thirst)".
# Each entry: lay term or phrase → expanded medical vocabulary string.
MEDICAL_SYNONYMS: Dict[str, str] = {
    # Hyperglycemia / thirst / urination
    "thirsty": "thirst polydipsia excessive thirst drinking lots water",
    "always thirsty": "thirst polydipsia excessive thirst drinking lots water",
    "drinking lots": "thirst polydipsia excessive thirst drinking lots water",
    "keep drinking": "thirst polydipsia excessive thirst",
    "peeing": "urination frequent urination polyuria bathroom",
    "pee a lot": "urination frequent urination polyuria nocturia",
    "urinate often": "urination frequent urination polyuria",
    "bathroom often": "urination frequent urination polyuria nocturia",
    "wake up to pee": "urination frequent urination polyuria nocturia",
    "very hungry": "hunger polyphagia excessive hunger appetite",
    "always hungry": "hunger polyphagia excessive hunger appetite",
    "losing weight": "weight loss unexplained weight loss",
    "lost weight": "weight loss unexplained weight loss",
    "losing weight without trying": "unexplained weight loss",
    "tired": "fatigue low energy weakness exhaustion",
    "exhausted": "fatigue low energy weakness exhaustion lethargy",
    "no energy": "fatigue low energy weakness exhaustion",
    "always tired": "fatigue low energy weakness exhaustion",
    "blurry vision": "blurred fluctuating vision eyes sight",
    "blurry eyes": "blurred fluctuating vision eyes sight",
    "vision problems": "blurred fluctuating vision eyes",
    "cant see clearly": "blurred fluctuating vision",
    "wounds not healing": "slow healing wounds cuts sores",
    "cut won't heal": "slow healing wounds cuts sores",
    "slow healing": "slow healing wounds cuts sores",
    "frequent infections": "recurrent infections skin uti yeast",
    "keep getting sick": "recurrent infections skin uti yeast",
    "sweet breath": "fruity acetone breath ketones",
    "fruity breath": "fruity acetone breath ketones ketoacidosis",
    "dry mouth": "dry mouth thirst",
    "headache": "headaches head pain",

    # Hypoglycemia
    "shaking": "shakiness trembling tremor hypoglycemia",
    "shaky": "shakiness trembling tremor hypoglycemia",
    "hands shaking": "shakiness trembling tremor",
    "sweating": "sweating without exertion perspiration",
    "sweating a lot": "sweating without exertion perspiration",
    "heart racing": "palpitations rapid heartbeat tachycardia",
    "heart pounding": "palpitations rapid heartbeat tachycardia",
    "anxious": "anxiety irritability sudden anxiety",
    "irritable": "sudden anxiety irritability",
    "confused": "confusion difficulty concentrating mental clarity",
    "can't concentrate": "confusion difficulty concentrating brain fog",
    "dizzy": "dizziness lightheadedness vertigo",
    "lightheaded": "dizziness lightheadedness",
    "hungry after eating": "hunger after recent meal reactive hypoglycemia",
    "pale": "pallor pale skin",
    "weak": "weakness feeling faint low energy",
    "nightmares": "nightmares night sweats nocturnal hypoglycemia",
    "night sweats": "nightmares night sweats nocturnal",

    # Insulin resistance / pre-diabetes
    "belly fat": "central obesity abdominal fat visceral",
    "stomach fat": "central obesity abdominal fat visceral",
    "potbelly": "central obesity abdominal fat",
    "dark patches": "acanthosis nigricans dark skin patches neck armpits",
    "dark neck": "acanthosis nigricans dark skin patches neck",
    "dark armpits": "acanthosis nigricans dark skin patches armpits",
    "tired after eating": "fatigue after carbohydrate meals post-prandial",
    "tired after carbs": "fatigue after carbohydrate meals",
    "sleepy after eating": "fatigue after carbohydrate meals",
    "brain fog": "brain fog poor concentration cognitive",
    "forgetful": "brain fog poor concentration cognitive",
    "sugar cravings": "cravings sweets carbohydrates sugar",
    "craving sugar": "cravings sweets carbohydrates sugar",
    "high blood pressure": "elevated blood pressure hypertension",
    "skin tags": "skin tags acrochordon insulin resistance",

    # Type 1 / DKA
    "sudden thirst": "rapid onset thirst urination acute",
    "rapid thirst": "rapid onset thirst urination acute",
    "labored breathing": "deep laboured breathing kussmaul ketoacidosis",
    "deep breathing": "deep laboured breathing kussmaul",
    "vomiting": "severe nausea vomiting emesis",
    "stomach pain": "abdominal pain gastrointestinal",
    "extreme tiredness": "extreme fatigue lethargy",
    "passing out": "confusion altered consciousness syncope",

    # Neuropathy
    "tingling": "tingling numbness neuropathy feet hands peripheral",
    "numb feet": "numbness tingling neuropathy feet",
    "numb hands": "numbness tingling neuropathy hands",
    "burning feet": "burning pain lower extremities neuropathy",
    "feet burning": "burning pain lower extremities neuropathy",
    "electric shocks": "sharp electric shock pains legs neuropathy",
    "sensitive skin": "increased sensitivity touch allodynia",
    "foot ulcer": "foot ulcers sores not healing diabetic foot",
    "sore on foot": "foot ulcers sores not healing diabetic foot",
    "weak legs": "muscle weakness feet legs",

    # Autonomic neuropathy
    "dizzy standing up": "postural hypotension dizziness standing orthostatic",
    "dizzy when I stand": "postural hypotension dizziness standing orthostatic",
    "bloating": "gastroparesis early satiety bloating nausea",
    "feel full fast": "gastroparesis early satiety bloating",
    "bladder problems": "bladder dysfunction incomplete emptying incontinence",
    "can't empty bladder": "bladder dysfunction incomplete emptying",
    "sweating problems": "excessive reduced sweating autonomic",
    "fast heartbeat": "resting tachycardia rapid heart rate",
    "no hypoglycemia warning": "hypoglycemia unawareness autonomic neuropathy",
}

# All canonical symptom strings from both maps — the "vocabulary" of the matcher
_ALL_CANONICAL: List[str] = list(dict.fromkeys(
    s for symptoms in _CONDITION_SYMPTOM_MAP.values() for s in symptoms
))


class SemanticSymptomMatcher:
    """
    Matches free-text patient input to canonical symptom strings using:
    1. Exact match check (fastest, highest confidence)
    2. Medical synonym expansion → TF-IDF character n-gram cosine similarity

    No model downloads required. Works fully offline.
    """

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold
        self._canonical = _ALL_CANONICAL
        self._expanded_canonical = [self._expand(s) for s in self._canonical]
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )
        # Fit vectorizer on all canonical texts
        self._vectorizer.fit(self._expanded_canonical)
        self._canonical_matrix = self._vectorizer.transform(self._expanded_canonical)
        logger.info(f"SemanticSymptomMatcher ready — {len(self._canonical)} canonical symptoms")

    def _expand(self, text: str) -> str:
        """Expand text with medical synonyms to improve matching coverage."""
        text_lower = text.lower()
        expanded = text_lower
        for key, expansion in MEDICAL_SYNONYMS.items():
            if key in text_lower:
                expanded += " " + expansion
        return expanded

    def match(self, user_input: str) -> Tuple[Optional[str], float]:
        """
        Match one user input string to the best canonical symptom.

        Returns:
            (canonical_symptom, similarity_score) or (None, 0.0) if below threshold
        """
        # Step 1: exact match — if the user selected a checkbox, it already is canonical
        if user_input in self._canonical:
            return user_input, 1.0

        # Step 2: case-insensitive exact match
        user_lower = user_input.lower().strip()
        for c in self._canonical:
            if c.lower() == user_lower:
                return c, 1.0

        # Step 3: TF-IDF semantic match on expanded text
        expanded = self._expand(user_input)
        user_vec = self._vectorizer.transform([expanded])
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(user_vec, self._canonical_matrix).flatten()
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= self.threshold:
            return self._canonical[best_idx], best_score
        return None, best_score

    def match_all(self, user_inputs: List[str]) -> Dict[str, Any]:
        """
        Match a list of user inputs.

        Returns:
            {
              "matched":   [(user_input, canonical, score), ...],
              "unmatched": [(user_input, best_score), ...],
              "canonical_list": [canonical strings that matched, deduplicated]
            }
        """
        matched = []
        unmatched = []
        seen_canonical = set()
        canonical_list = []

        for user_input in user_inputs:
            canonical, score = self.match(user_input)
            if canonical and canonical not in seen_canonical:
                matched.append((user_input, canonical, score))
                seen_canonical.add(canonical)
                canonical_list.append(canonical)
            elif canonical and canonical in seen_canonical:
                # Already matched — still record as matched, just don't duplicate
                matched.append((user_input, canonical, score))
            else:
                unmatched.append((user_input, score))

        return {
            "matched":        matched,
            "unmatched":      unmatched,
            "canonical_list": canonical_list,
        }


# Module-level singleton — created once, reused for all calls
_matcher = SemanticSymptomMatcher(threshold=0.35)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — PIMA INDIANS DIABETES DATASET + LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

# Pima Indians Diabetes Dataset (262 rows)
# Source: UCI Machine Learning Repository (public domain)
# Citation: Smith, J.W. et al. (1988). Using the ADAP learning algorithm to
#           forecast the onset of diabetes mellitus. Proceedings of the Symposium
#           on Computer Applications in Medical Care, pp. 261-265.
# 768 female patients of Pima Indian heritage, aged >= 21.
# Features: pregnancies, glucose, blood_pressure, skin_thickness, insulin,
#           bmi, diabetes_pedigree_function, age
# Target: outcome (1 = diabetes, 0 = no diabetes)

_PIMA_DATA = [
    [6,148,72,35,0,33.6,0.627,50,1],[1,85,66,29,0,26.6,0.351,31,0],
    [8,183,64,0,0,23.3,0.672,32,1],[1,89,66,23,94,28.1,0.167,21,0],
    [0,137,40,35,168,43.1,2.288,33,1],[5,116,74,0,0,25.6,0.201,30,0],
    [3,78,50,32,88,31.0,0.248,26,1],[10,115,0,0,0,35.3,0.134,29,0],
    [2,197,70,45,543,30.5,0.158,53,1],[8,125,96,0,0,0.0,0.232,54,1],
    [4,110,92,0,0,37.6,0.191,30,0],[10,168,74,0,0,38.0,0.537,34,1],
    [10,139,80,0,0,27.1,1.441,57,0],[1,189,60,23,846,30.1,0.398,59,1],
    [5,166,72,19,175,25.8,0.587,51,1],[7,100,0,0,0,30.0,0.484,32,1],
    [0,118,84,47,230,45.8,0.551,31,1],[7,107,74,0,0,29.6,0.254,31,1],
    [1,103,30,38,83,43.3,0.183,33,0],[1,115,70,30,96,34.6,0.529,32,1],
    [3,126,88,41,235,39.3,0.704,27,0],[8,99,84,0,0,35.4,0.388,50,0],
    [7,196,90,0,0,39.8,0.451,41,1],[9,119,80,35,0,29.0,0.263,29,1],
    [11,143,94,33,146,36.6,0.254,51,1],[10,125,70,26,115,31.1,0.205,41,1],
    [7,147,76,0,0,39.4,0.257,43,1],[1,97,66,15,140,23.2,0.487,22,0],
    [13,145,82,19,110,22.2,0.245,57,0],[5,109,75,26,0,36.0,0.546,60,0],
    [3,158,76,36,245,31.6,0.851,28,1],[3,88,58,11,54,24.8,0.267,22,0],
    [6,92,92,0,0,19.9,0.188,28,0],[10,122,78,31,0,27.6,0.512,45,0],
    [4,103,60,33,192,24.0,0.966,33,0],[11,138,76,0,0,33.2,0.420,35,0],
    [9,102,76,37,0,32.9,0.665,46,1],[2,90,68,42,0,38.2,0.503,27,1],
    [4,111,72,47,207,37.1,1.390,56,1],[3,180,64,25,70,34.0,0.271,26,0],
    [7,133,84,0,0,40.2,0.696,37,0],[7,106,92,18,0,22.7,0.235,48,0],
    [9,171,110,24,240,45.4,0.721,54,1],[7,159,64,0,0,27.4,0.294,40,0],
    [0,180,66,39,0,42.0,1.893,25,1],[1,146,56,0,0,29.7,0.564,29,0],
    [2,71,70,27,0,28.0,0.586,22,0],[7,103,66,32,0,39.1,0.344,31,1],
    [7,105,0,0,0,0.0,0.305,24,0],[1,103,80,11,82,19.4,0.491,22,0],
    [1,101,50,15,36,24.2,0.526,26,0],[5,88,66,21,23,24.4,0.342,30,0],
    [8,176,90,34,300,33.7,0.467,58,1],[7,150,66,42,342,34.7,0.718,42,0],
    [1,73,50,10,0,23.0,0.248,21,0],[7,187,68,39,304,37.7,0.254,41,1],
    [0,100,88,60,110,46.8,0.962,31,0],[0,146,82,0,0,40.5,1.781,44,0],
    [0,105,64,41,142,41.5,0.173,22,0],[2,84,0,0,0,0.0,0.304,21,0],
    [8,133,72,0,0,32.9,0.270,39,1],[5,44,62,0,0,25.0,0.587,36,0],
    [2,141,58,34,128,25.4,0.699,24,0],[7,114,66,0,0,32.8,0.258,42,1],
    [5,99,74,27,0,29.0,0.203,32,0],[0,109,88,30,0,32.5,0.855,38,1],
    [2,109,92,0,0,42.7,0.845,54,0],[1,95,66,13,38,19.6,0.334,25,0],
    [4,146,85,27,100,28.9,0.189,27,0],[8,129,110,46,130,67.1,0.319,31,1],
    [0,102,86,17,105,29.3,0.695,27,0],[8,154,78,32,0,32.4,0.443,45,1],
    [1,87,60,37,75,37.2,0.509,22,0],[6,129,90,7,326,19.6,0.582,60,0],
    [4,129,86,20,270,35.1,0.231,23,0],[6,92,62,32,126,32.0,0.085,46,0],
    [8,133,0,0,0,0.0,0.640,38,1],[9,124,70,33,402,35.4,0.282,34,0],
    [4,99,72,17,0,25.6,0.294,28,0],[0,101,62,0,0,21.9,0.336,25,0],
    [3,128,78,0,0,21.1,0.268,55,0],[2,74,0,0,0,0.0,0.102,22,0],
    [7,83,78,26,71,29.3,0.767,36,0],[0,101,76,0,0,19.4,0.280,28,0],
    [2,122,70,27,0,36.8,0.340,27,0],[5,121,72,23,112,26.2,0.245,30,0],
    [1,126,60,0,0,30.1,0.349,47,1],[1,93,70,31,0,30.4,0.315,23,0],
    [0,126,84,29,215,30.7,0.520,24,0],[3,80,0,0,0,0.0,0.174,22,0],
    [5,106,82,30,0,39.5,0.286,38,0],[2,155,52,27,540,38.7,0.240,25,1],
    [4,113,30,13,269,46.1,0.209,19,0],[0,100,70,26,50,30.8,0.597,21,0],
    [4,136,70,0,0,31.2,1.182,22,1],[5,168,64,0,0,32.9,0.135,41,1],
    [2,123,48,32,165,42.1,0.520,26,0],[4,115,72,0,0,28.9,0.376,46,1],
    [3,111,56,39,0,30.1,0.557,30,0],[3,196,70,27,100,36.5,0.425,24,1],
    [0,174,69,0,0,43.9,0.702,21,1],[6,105,80,28,0,32.5,0.878,26,0],
    [5,114,74,0,0,24.9,0.744,57,0],[4,129,60,12,231,27.5,0.527,31,0],
    [3,125,58,0,0,31.6,0.151,24,0],[8,188,78,0,0,47.9,0.137,43,1],
    [2,108,64,0,0,30.8,0.158,21,0],[1,99,58,10,0,25.4,0.551,21,0],
    [6,103,72,32,190,37.7,0.324,55,0],[0,124,56,13,105,21.8,0.452,21,0],
    [6,131,0,0,0,0.0,0.422,27,1],[7,116,0,0,0,23.4,0.918,45,1],
    [3,99,62,19,110,36.8,0.589,28,0],[2,100,66,20,90,32.9,0.867,28,1],
    [0,95,64,39,105,44.6,0.366,22,0],[0,105,90,0,0,29.6,0.197,46,0],
    [6,183,94,0,0,40.8,1.461,45,0],[0,126,78,38,140,36.2,0.720,28,0],
    [3,153,0,0,0,0.0,0.428,23,0],[4,147,74,25,475,41.4,0.385,51,1],
    [5,154,78,46,520,36.4,0.389,28,1],[6,111,64,39,0,34.2,0.260,24,0],
    [1,75,50,32,0,23.2,0.539,21,0],[1,99,66,15,36,19.6,0.510,28,0],
    [9,99,92,25,80,25.1,0.432,32,1],[5,80,76,41,88,32.9,0.780,35,0],
    [1,175,90,32,107,42.5,0.737,32,1],[6,185,92,0,0,28.1,0.276,35,0],
    [5,99,54,28,83,34.0,0.499,30,0],[1,99,72,30,18,38.6,0.412,21,0],
]

_PIMA_COLS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "pedigree", "age", "outcome",
]

_PIMA_FEATURE_COLS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "pedigree", "age",
]


class PimaModel:
    """
    Logistic Regression trained on the Pima Indians Diabetes Dataset.

    Predicts diabetes probability from: pregnancies, glucose, blood_pressure,
    skin_thickness, insulin, bmi, pedigree_function, age.

    Accuracy metrics are computed via 5-fold stratified cross-validation and
    a held-out 20% test set. Results are stored in self.metrics.
    """

    def __init__(self):
        self._model: Optional[LogisticRegression] = None
        self._scaler: Optional[StandardScaler] = None
        self.metrics: Dict[str, Any] = {}
        self._trained = False
        self._train()

    def _build_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(_DATASET_PATH)
        # Replace biologically impossible zeros with NaN, then impute with median
        for col in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
            df[col] = df[col].replace(0, np.nan)
        df = df.fillna(df.median(numeric_only=True))
        return df

    def _train(self):
        """Train model and compute all accuracy metrics."""
        if _MODEL_CACHE_PATH.exists():
            try:
                with open(_MODEL_CACHE_PATH, "rb") as f:
                    saved = pickle.load(f)
                self._model   = saved["model"]
                self._scaler  = saved["scaler"]
                self.metrics  = saved["metrics"]
                self._trained = True
                logger.info("PimaModel loaded from cache")
                return
            except Exception:
                pass  # re-train if cache is corrupt

        logger.info("PimaModel: training on Pima Indians Diabetes Dataset...")
        df = self._build_dataframe()
        X = df[_PIMA_FEATURE_COLS].values
        y = df["outcome"].values

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Train / test split (80/20, stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.20, random_state=42, stratify=y
        )

        self._model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        self._model.fit(X_train, y_train)

        # Hold-out test metrics
        y_pred = self._model.predict(X_test)
        y_prob = self._model.predict_proba(X_test)[:, 1]
        cm     = confusion_matrix(y_test, y_pred)

        # 5-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_acc = cross_val_score(self._model, X_scaled, y, cv=cv, scoring="accuracy")
        cv_f1  = cross_val_score(self._model, X_scaled, y, cv=cv, scoring="f1")
        cv_auc = cross_val_score(self._model, X_scaled, y, cv=cv, scoring="roc_auc")

        # Feature importance from logistic regression coefficients
        feature_importance = sorted(
            zip(_PIMA_FEATURE_COLS, self._model.coef_[0].tolist()),
            key=lambda x: abs(x[1]), reverse=True
        )

        self.metrics = {
            # Dataset info
            "dataset":          "Pima Indians Diabetes Dataset (UCI ML Repository)",
            "dataset_citation": "Smith et al. (1988), ADAP algorithm, SCAMC pp.261-265",
            "total_samples":    len(df),
            "diabetic_samples": int(y.sum()),
            "non_diabetic_samples": int((y == 0).sum()),
            "train_samples":    len(X_train),
            "test_samples":     len(X_test),
            "features_used":    _PIMA_FEATURE_COLS,

            # Hold-out test metrics
            "test_accuracy":    round(float(accuracy_score(y_test, y_pred)), 4),
            "test_precision":   round(float(precision_score(y_test, y_pred)), 4),
            "test_recall":      round(float(recall_score(y_test, y_pred)), 4),
            "test_f1":          round(float(f1_score(y_test, y_pred)), 4),
            "test_roc_auc":     round(float(roc_auc_score(y_test, y_prob)), 4),
            "confusion_matrix": {
                "true_negative":  int(cm[0, 0]),
                "false_positive": int(cm[0, 1]),
                "false_negative": int(cm[1, 0]),
                "true_positive":  int(cm[1, 1]),
            },

            # Cross-validation metrics (more reliable)
            "cv_folds":         5,
            "cv_accuracy":      round(float(cv_acc.mean()), 4),
            "cv_accuracy_std":  round(float(cv_acc.std()), 4),
            "cv_f1":            round(float(cv_f1.mean()), 4),
            "cv_f1_std":        round(float(cv_f1.std()), 4),
            "cv_roc_auc":       round(float(cv_auc.mean()), 4),
            "cv_roc_auc_std":   round(float(cv_auc.std()), 4),

            # Feature importance
            "feature_importance": [
                {"feature": f, "coefficient": round(c, 4)}
                for f, c in feature_importance
            ],

            # Classification report
            "classification_report": classification_report(
                y_test, y_pred, target_names=["No Diabetes", "Diabetes"]
            ),
        }

        self._trained = True

        # Cache to disk
        try:
            with open(_MODEL_CACHE_PATH, "wb") as f:
                pickle.dump({
                    "model": self._model,
                    "scaler": self._scaler,
                    "metrics": self.metrics,
                }, f)
            logger.info("PimaModel cached to disk")
        except Exception as e:
            logger.warning(f"Could not cache model: {e}")

        logger.info(
            f"PimaModel trained — "
            f"CV Accuracy: {self.metrics['cv_accuracy']:.2%} ± {self.metrics['cv_accuracy_std']:.2%} | "
            f"ROC-AUC: {self.metrics['cv_roc_auc']:.4f}"
        )

    def predict(self, lab_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict diabetes probability for a single patient.

        Args:
            lab_values: dict with any subset of:
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, pedigree, age

        Returns:
            {
              "probability":  float (0-1),
              "prediction":   "Diabetes" | "No Diabetes",
              "confidence":   "High" | "Moderate" | "Low",
              "risk_level":   "Critical" | "High" | "Moderate" | "Low",
              "model":        "LogisticRegression (Pima Dataset)"
            }
        """
        if not self._trained or self._model is None:
            return {"error": "Model not trained", "probability": None}

        # Build feature vector — use 0 for missing features (imputed by median during training)
        df_input = pd.DataFrame([{
            col: lab_values.get(col, 0) for col in _PIMA_FEATURE_COLS
        }])

        # Replace impossible zeros with dataset medians for the 5 imputed columns
        df_source = self._build_dataframe()
        for col in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
            if df_input[col].iloc[0] == 0:
                df_input[col] = df_source[col].median()

        X = self._scaler.transform(df_input[_PIMA_FEATURE_COLS].values)
        prob = float(self._model.predict_proba(X)[0, 1])
        prediction = "Diabetes" if prob >= 0.50 else "No Diabetes"

        if prob >= 0.80:
            confidence, risk_level = "High",     "Critical"
        elif prob >= 0.65:
            confidence, risk_level = "High",     "High"
        elif prob >= 0.40:
            confidence, risk_level = "Moderate", "Moderate"
        else:
            confidence, risk_level = "High",     "Low"

        return {
            "probability":  round(prob, 4),
            "probability_pct": f"{prob * 100:.1f}%",
            "prediction":   prediction,
            "confidence":   confidence,
            "risk_level":   risk_level,
            "model":        "LogisticRegression (Pima Indians Diabetes Dataset)",
        }

    def get_accuracy_report(self) -> str:
        """Return a formatted human-readable accuracy report."""
        m = self.metrics
        cm = m["confusion_matrix"]
        lines = [
            "=" * 60,
            "PIMA MODEL — ACCURACY REPORT",
            "=" * 60,
            f"Dataset:    {m['dataset']}",
            f"Citation:   {m['dataset_citation']}",
            f"Samples:    {m['total_samples']} total "
            f"({m['diabetic_samples']} diabetic, {m['non_diabetic_samples']} non-diabetic)",
            f"Features:   {', '.join(m['features_used'])}",
            "",
            "── HOLD-OUT TEST SET (20%) ──────────────────────────",
            f"Accuracy:   {m['test_accuracy']:.4f}  ({m['test_accuracy']*100:.2f}%)",
            f"Precision:  {m['test_precision']:.4f}",
            f"Recall:     {m['test_recall']:.4f}",
            f"F1 Score:   {m['test_f1']:.4f}",
            f"ROC-AUC:    {m['test_roc_auc']:.4f}",
            "",
            "Confusion Matrix:",
            f"  True Negative  (no diabetes, correct): {cm['true_negative']}",
            f"  False Positive (no diabetes, wrong):   {cm['false_positive']}",
            f"  False Negative (diabetes, missed):     {cm['false_negative']}",
            f"  True Positive  (diabetes, correct):    {cm['true_positive']}",
            "",
            "── 5-FOLD CROSS-VALIDATION ──────────────────────────",
            f"CV Accuracy: {m['cv_accuracy']:.4f} ± {m['cv_accuracy_std']:.4f}",
            f"CV F1:       {m['cv_f1']:.4f} ± {m['cv_f1_std']:.4f}",
            f"CV ROC-AUC:  {m['cv_roc_auc']:.4f} ± {m['cv_roc_auc_std']:.4f}",
            "",
            "── FEATURE IMPORTANCE (by coefficient magnitude) ────",
        ]
        for fi in m["feature_importance"]:
            bar = "█" * int(abs(fi["coefficient"]) * 10)
            lines.append(f"  {fi['feature']:20s} {fi['coefficient']:+.4f}  {bar}")
        lines.append("")
        lines.append("── CLASSIFICATION REPORT ────────────────────────────")
        lines.append(m["classification_report"])
        lines.append("=" * 60)
        return "\n".join(lines)


# Module-level singleton — trained once, reused for all calls
_pima_model = PimaModel()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — LANGCHAIN TOOL (now uses semantic matching)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def map_symptoms_to_conditions(symptoms: List[str]) -> Dict[str, Any]:
    """
    Map patient symptoms to ranked diabetes condition hypotheses.

    Accepts BOTH canonical symptom strings (from checkboxes) AND
    free-text paraphrases (e.g. "always thirsty", "feet tingling").
    Semantic matching via TF-IDF + medical synonym expansion.

    Returns condition_hypotheses (sorted by match_count), top_hypothesis,
    total_symptoms_checked, unmatched_symptoms, and semantic_matches.
    """
    if not symptoms:
        return {
            "condition_hypotheses":   [],
            "top_hypothesis":         "No symptoms provided",
            "total_symptoms_checked": 0,
            "unmatched_symptoms":     [],
            "semantic_matches":       [],
        }

    # Step 1: semantic resolution — convert all inputs to canonical form
    match_result    = _matcher.match_all(symptoms)
    canonical_list  = match_result["canonical_list"]
    semantic_log    = [
        {"input": m[0], "matched_to": m[1], "score": round(m[2], 3)}
        for m in match_result["matched"]
    ]
    unmatched_inputs = [u[0] for u in match_result["unmatched"]]

    # Step 2: condition scoring on resolved canonical symptoms
    scored: Dict[str, Dict] = {}
    all_mapped: set = set()

    for condition, cond_syms in _CONDITION_SYMPTOM_MAP.items():
        matched = [s for s in canonical_list if s in cond_syms]
        if matched:
            scored[condition] = {
                "matched_symptoms": matched,
                "match_count":      len(matched),
                "coverage":         f"{len(matched)}/{len(cond_syms)}",
            }
            all_mapped.update(matched)

    ranked    = sorted(scored.items(), key=lambda x: x[1]["match_count"], reverse=True)
    unmatched = [s for s in canonical_list if s not in all_mapped] + unmatched_inputs

    return {
        "condition_hypotheses":   [{"condition": k, **v} for k, v in ranked],
        "top_hypothesis":         ranked[0][0] if ranked else "No condition mapped",
        "total_symptoms_checked": len(symptoms),
        "unmatched_symptoms":     unmatched,
        "semantic_matches":       semantic_log,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — CONTEXT BUILDER (unchanged from v2)
# ══════════════════════════════════════════════════════════════════════════════

def build_context(
    report_context: Optional[Dict] = None,
    risk_context:   Optional[Dict] = None,
    manual_text:    Optional[str]  = None,
    pima_result:    Optional[Dict] = None,
) -> str:
    """
    Normalise upstream agent outputs, Pima prediction, and free text
    into a single prompt string for DeepSeek-R1.
    """
    parts: List[str] = []

    if pima_result and not pima_result.get("error"):
        parts.append("=== PIMA DATASET PREDICTION ===")
        parts.append(f"  Diabetes Probability: {pima_result.get('probability_pct','N/A')}")
        parts.append(f"  Prediction: {pima_result.get('prediction','N/A')}")
        parts.append(f"  Risk Level: {pima_result.get('risk_level','N/A')}")
        parts.append(f"  Model: {pima_result.get('model','N/A')}")

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
        for key, label in [
            ("risk_tier", "Risk Tier"), ("risk_score", "Risk Score"),
            ("risk_level", "Risk Level"), ("risk_probability", "Risk Probability"),
        ]:
            if risk_context.get(key):
                parts.append(f"  {label}: {risk_context[key]}")
        if risk_context.get("ada_classifications"):
            parts.append(f"  ADA: {'; '.join(risk_context['ada_classifications'])}")
        if risk_context.get("recommended_action"):
            parts.append(f"  Recommended Action: {risk_context['recommended_action']}")

    if manual_text and manual_text.strip():
        parts.append("=== ADDITIONAL PATIENT NOTES ===")
        parts.append(f"  {manual_text.strip()}")

    return "\n".join(parts) if parts else "No upstream context provided."


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — LLM COMPONENTS (unchanged from v2)
# ══════════════════════════════════════════════════════════════════════════════

def _clean_llm_output(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,3}\s+", "", text, flags=re.MULTILINE)
    return text.strip()


SYMPTOM_REASONING_PROMPT = PromptTemplate(
    input_variables=[
        "upstream_context", "symptom_map", "symptoms_text",
        "top_hypothesis", "unmatched_symptoms", "pima_probability",
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

Pima dataset ML prediction: {pima_probability}

=== YOUR TASK ===
Write a focused diabetes symptom-analysis note. Use EXACTLY this structure, no more than 200 words total:

1. SYMPTOM INTERPRETATION: Which symptoms are most clinically significant for diabetes and why.
2. CONDITION CORRELATION: How the symptoms align with the upstream lab/risk findings and ML prediction.
3. PRIMARY HYPOTHESIS: Most likely diabetes type or condition based on the full picture.
4. DIFFERENTIAL: 1-2 alternative diabetes conditions the unmatched or overlapping symptoms may suggest.
5. CLINICAL RECOMMENDATION: Specific next steps — which specialist, which tests, which lifestyle or medication changes.

Be concise and clinically precise. Do not repeat lab values verbatim -- refer to them by finding.
Do not add AI disclaimers or suggest consulting a doctor generically -- give specific specialty referrals.
Do not use markdown formatting, headers, or bullet symbols. Write in plain numbered sections only.""",
)


def _build_chain():
    try:
        resp      = requests.get(OLLAMA_TAGS, timeout=4)
        available = [m["name"] for m in resp.json().get("models", [])]
        model_id  = next(
            (a for a in available if REASONING_MODEL["id"].split(":")[0] in a), None
        )
        if not model_id:
            logger.warning(f"DeepSeek-R1 not found. Run: {REASONING_MODEL['pull_cmd']}")
            return None

        llm   = Ollama(model=model_id, temperature=0.2, num_predict=1200)
        chain = SYMPTOM_REASONING_PROMPT | llm | StrOutputParser()
        logger.info(f"SymptomAnalyzerAgent using: {model_id}")
        return chain
    except Exception as e:
        logger.warning(f"Ollama unavailable: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — RULE-BASED FALLBACK (unchanged from v2)
# ══════════════════════════════════════════════════════════════════════════════

def _rule_based_reasoning(
    symptom_map_result: Dict,
    upstream_context:   str,
    symptoms:           List[str],
    pima_result:        Optional[Dict] = None,
) -> str:
    hyps      = symptom_map_result.get("condition_hypotheses", [])
    top       = symptom_map_result.get("top_hypothesis", "")
    unmatched = symptom_map_result.get("unmatched_symptoms", [])
    lines     = []

    if not symptoms:
        lines.append("No symptoms were provided.")
        if upstream_context and "No upstream context" not in upstream_context:
            lines.append("\nReasoning is based on upstream agent context only:")
            lines.append(upstream_context)
        lines.append("\nRecommendation: Select symptoms from the checklist or enter a text description.")
        return "\n".join(lines)

    lines.append(f"SYMPTOM INTERPRETATION ({len(symptoms)} reported)")
    for s in symptoms:
        lines.append(f"  - {s}")

    # Show semantic resolution if any free-text was matched
    sem = symptom_map_result.get("semantic_matches", [])
    free_text_matches = [m for m in sem if m["score"] < 1.0]
    if free_text_matches:
        lines.append("\nSemantic matches resolved:")
        for m in free_text_matches:
            lines.append(f"  '{m['input']}' → '{m['matched_to']}' (score {m['score']:.2f})")

    if hyps:
        lines.append(f"\nPRIMARY HYPOTHESIS: {top}")
        lines.append(f"  Matched: {', '.join(hyps[0]['matched_symptoms'])}")
        lines.append(f"  Coverage: {hyps[0]['coverage']} known symptoms for this condition")
        if len(hyps) > 1:
            lines.append("\nDIFFERENTIAL:")
            for h in hyps[1:3]:
                lines.append(f"  - {h['condition']} — {h['match_count']} symptom(s) matched")
    else:
        lines.append("\nNo symptoms matched any known diabetes condition pattern.")
        lines.append("Consider broader diabetes workup or specialist evaluation.")

    if unmatched:
        lines.append(f"\nUNMATCHED SYMPTOMS: {', '.join(unmatched)}")
        lines.append("  These may indicate non-diabetes causes — consider broader evaluation.")

    if pima_result and not pima_result.get("error"):
        lines.append(f"\nML PREDICTION (Pima Dataset):")
        lines.append(f"  Probability: {pima_result.get('probability_pct','N/A')}")
        lines.append(f"  Prediction:  {pima_result.get('prediction','N/A')}")
        lines.append(f"  Risk Level:  {pima_result.get('risk_level','N/A')}")

    if upstream_context and "No upstream context" not in upstream_context:
        lines.append("\nUPSTREAM CONTEXT SUMMARY:")
        for line in upstream_context.split("\n"):
            if any(kw in line for kw in ["Flag", "Risk", "ADA", "Abnormal", "Probability"]):
                lines.append(f"  {line.strip()}")

    recs = {
        "Type 2 Diabetes":                "Refer to Endocrinologist. Confirm with repeat HbA1c + OGTT. Initiate MNT and consider Metformin.",
        "Type 1 Diabetes":                "Urgent referral to Endocrinologist. Check C-peptide, GAD antibodies, fasting insulin. Initiate insulin therapy.",
        "Pre-diabetes / Insulin Resistance": "Lifestyle intervention program. Repeat HbA1c in 3 months. Screen for metabolic syndrome.",
        "Gestational Diabetes":           "Refer to OB-GYN + Endocrinologist. OGTT immediately. Monitor fetal growth. Dietary modification first line.",
        "Hypoglycemia":                   "Evaluate for reactive vs. fasting hypoglycemia. 72-hour fast test if indicated. Review current medications.",
        "Diabetic Ketoacidosis (DKA)":    "EMERGENCY — immediate hospital admission. IV fluids, insulin drip, electrolyte correction. Monitor hourly.",
        "Diabetic Peripheral Neuropathy": "Refer to Neurologist. Monofilament foot exam. Optimize glycemic control. Consider gabapentin or duloxetine.",
        "Diabetic Autonomic Neuropathy":  "Refer to Neurologist + Gastroenterologist. Orthostatic BP measurement. HRV testing. Optimize glycemic control.",
    }
    lines.append(f"\nRECOMMENDATION: {recs.get(top, 'Refer to Endocrinologist for comprehensive diabetes evaluation.')}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def analyze_symptoms(
    symptoms:       List[str],
    lab_values:     Optional[Dict[str, float]] = None,
    report_context: Optional[Dict] = None,
    risk_context:   Optional[Dict] = None,
    manual_text:    Optional[str]  = None,
    use_llm:        bool           = True,
) -> Dict[str, Any]:
    """
    Run the full diabetes symptom analysis pipeline (v3).

    Args:
        symptoms:       Symptom strings — checkboxes OR free-text paraphrases.
                        e.g. ["always thirsty", "feet tingling", "blurry eyes"]
        lab_values:     Optional lab dict for Pima ML prediction.
                        Keys: pregnancies, glucose, blood_pressure, skin_thickness,
                              insulin, bmi, pedigree, age
        report_context: Output dict from ReportAnalyzerAgent (optional).
        risk_context:   Output dict from RiskPredictorAgent (optional).
        manual_text:    Free-text notes from user (optional).
        use_llm:        Set False to force rule-based mode.

    Returns:
        {
          "symptom_mapping":  { condition_hypotheses, top_hypothesis,
                                unmatched_symptoms, semantic_matches },
          "pima_prediction":  { probability, prediction, risk_level, ... },
          "pima_metrics":     { accuracy, precision, recall, f1, roc_auc, ... },
          "reasoning":        str,
          "model_used":       "deepseek-r1:8b" | "rule-based",
          "input_summary":    { symptoms_count, sources_used, top_hypothesis },
        }
    """
    # Step A — semantic symptom mapping
    symptom_map_result = map_symptoms_to_conditions.invoke({"symptoms": symptoms})

    # Step B — Pima ML prediction (if lab values provided)
    pima_result: Optional[Dict] = None
    if lab_values:
        pima_result = _pima_model.predict(lab_values)
    elif report_context:
        # Try to extract lab values from report_context automatically
        params = report_context.get("parameters", {})
        auto_labs: Dict[str, float] = {}
        field_map = {
            "glucose": "glucose", "bmi": "bmi", "age": "age",
            "pregnancies": "pregnancies", "blood_pressure": "blood_pressure",
            "skin_thickness": "skin_thickness", "insulin": "insulin",
        }
        for src_key, dst_key in field_map.items():
            if src_key in params:
                val = params[src_key]
                auto_labs[dst_key] = float(val["value"]) if isinstance(val, dict) else float(val)
        if auto_labs:
            pima_result = _pima_model.predict(auto_labs)

    # Step C — context assembly (includes pima_result)
    upstream_context = build_context(report_context, risk_context, manual_text, pima_result)

    # Prepare prompt variables
    top_hyp   = symptom_map_result.get("top_hypothesis", "No condition mapped")
    hyps      = symptom_map_result.get("condition_hypotheses", [])
    unmatched = symptom_map_result.get("unmatched_symptoms", [])

    symp_text = "\n".join(f"  - {s}" for s in symptoms) if symptoms else "  None provided."
    hyp_text  = "\n".join(
        f"  - {h['condition']}: {h['match_count']} symptom(s) matched "
        f"({', '.join(h['matched_symptoms'][:3])}{'...' if len(h['matched_symptoms']) > 3 else ''})"
        for h in hyps[:5]
    ) or "  No condition matches found."
    unmatched_text = ", ".join(unmatched) if unmatched else "None"
    pima_prob_text = (
        f"{pima_result.get('probability_pct','N/A')} ({pima_result.get('prediction','N/A')})"
        if pima_result and not pima_result.get("error")
        else "Not available (no lab values provided)"
    )

    # Step D — reasoning (LLM or rule-based)
    reasoning  = ""
    model_used = "rule-based"
    fallback   = _rule_based_reasoning(symptom_map_result, upstream_context, symptoms, pima_result)

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
                    "pima_probability":   pima_prob_text,
                })
                llm_output = _clean_llm_output(llm_output if isinstance(llm_output, str) else "")
                if llm_output and len(llm_output.strip()) > 80:
                    reasoning  = llm_output.strip()
                    model_used = REASONING_MODEL["id"]
                    logger.info(f"Reasoning via {REASONING_MODEL['name']}")
                else:
                    logger.warning("LLM output too short — using rule-based fallback")
            except Exception as e:
                logger.warning(f"LLM chain failed: {e} — using rule-based fallback")

    if not reasoning or not reasoning.strip():
        reasoning = fallback

    sources_used = []
    if report_context: sources_used.append("ReportAnalyzerAgent")
    if risk_context:   sources_used.append("RiskPredictorAgent")
    if lab_values:     sources_used.append("PimaLabValues")
    if manual_text:    sources_used.append("ManualText")
    if symptoms:       sources_used.append("SymptomCheckboxes")

    return {
        "symptom_mapping": symptom_map_result,
        "pima_prediction": pima_result,
        "pima_metrics":    _pima_model.metrics,
        "reasoning":       reasoning,
        "model_used":      model_used,
        "input_summary": {
            "symptoms_count": len(symptoms),
            "sources_used":   sources_used,
            "top_hypothesis": top_hyp,
        },
    }


def get_accuracy_report() -> str:
    """Return the full accuracy report for the Pima dataset model."""
    return _pima_model.get_accuracy_report()


__all__ = [
    "analyze_symptoms",
    "map_symptoms_to_conditions",
    "build_context",
    "get_accuracy_report",
    "SYMPTOM_CATEGORIES",
    "MEDICAL_SYNONYMS",
    "REASONING_MODEL",
    "SemanticSymptomMatcher",
    "PimaModel",
]


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("TEST 1 — ACCURACY REPORT (Pima Dataset)")
    print("=" * 60)
    print(get_accuracy_report())

    print("\n" + "=" * 60)
    print("TEST 2 — SEMANTIC MATCHING (free-text input)")
    print("=" * 60)
    free_text_symptoms = [
        "always thirsty",
        "peeing too much at night",
        "feet tingling and numb",
        "shaking hands in the morning",
        "my vision is blurry",
        "feel tired all the time",
        "belly fat",
        "dark patches on neck",
        "completely unrelated phrase",
    ]
    match_result = _matcher.match_all(free_text_symptoms)
    print("Semantic matches:")
    for m in match_result["matched"]:
        print(f"  {m[2]:.3f}  '{m[0]}' → '{m[1]}'")
    print("Unmatched:")
    for u in match_result["unmatched"]:
        print(f"  {u[1]:.3f}  '{u[0]}'")

    print("\n" + "=" * 60)
    print("TEST 3 — PIMA ML PREDICTION")
    print("=" * 60)
    lab_cases = [
        {"label": "High risk patient",    "values": {"glucose": 148, "bmi": 33.6, "age": 50, "pregnancies": 6}},
        {"label": "Low risk patient",     "values": {"glucose": 85,  "bmi": 26.6, "age": 31, "pregnancies": 1}},
        {"label": "Borderline patient",   "values": {"glucose": 120, "bmi": 29.0, "age": 40, "pregnancies": 3}},
    ]
    for case in lab_cases:
        pred = _pima_model.predict(case["values"])
        print(f"  {case['label']}: {pred['probability_pct']} → {pred['prediction']} ({pred['risk_level']})")

    print("\n" + "=" * 60)
    print("TEST 4 — FULL PIPELINE (free-text symptoms + lab values)")
    print("=" * 60)
    result = analyze_symptoms(
        symptoms   = free_text_symptoms[:7],
        lab_values = {"glucose": 148, "bmi": 29.1, "age": 45, "pregnancies": 3},
        manual_text = "Symptoms worsening over 3 months. Family history of Type 2 diabetes.",
        use_llm    = True,
    )
    print("\n-- Symptom mapping --")
    for h in result["symptom_mapping"]["condition_hypotheses"]:
        print(f"  {h['condition']}: {h['match_count']} matched ({h['coverage']})")
    print("\n-- Semantic resolution --")
    for m in result["symptom_mapping"].get("semantic_matches", []):
        if m["score"] < 1.0:
            print(f"  '{m['input']}' → '{m['matched_to']}' ({m['score']:.3f})")
    if result.get("pima_prediction"):
        p = result["pima_prediction"]
        print(f"\n-- Pima ML prediction --")
        print(f"  {p['probability_pct']} → {p['prediction']} ({p['risk_level']})")
    print(f"\n-- Model used: {result['model_used']} --")
    print(f"\n-- Clinical reasoning --\n{result['reasoning']}")
    print(f"\n-- Input summary --\n{json.dumps(result['input_summary'], indent=2)}")