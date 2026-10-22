"""Tests for the cardiovascular model: mapping, evaluation, and honesty rules.

These cover the rules the model follows:

* the dataset columns map onto the API's field names, and nothing else is used;
* ``smoker`` is not fabricated — the dataset has no smoking column, so the
  model does not use it and the score does not change with it;
* an unavailable feature is not a negative value;
* a missing required feature makes the model decline, not guess;
* the model reports the metrics it measured and the limitations it carries.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from Agents.plugins.cardio_model import (
    BASE_FEATURES,
    DATASET_ONLY_COLUMNS,
    DEFAULT_DATA_PATH,
    FEATURE_MAPPING,
    OPTIONAL_FEATURES,
    TARGET_COLUMN,
    UNAVAILABLE_FEATURES,
    VARIANTS,
    CardioModel,
)
from Agents.plugins.cardiovascular import CardiovascularPlugin

DATA_PATH = Path("data/Cardiovascular_Disease_Dataset.csv")
METADATA_PATH = Path("Agents/plugins/cardio_model_metadata.json")


@pytest.fixture(scope="module")
def dataset():
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def metadata():
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE MAPPING
# ═════════════════════════════════════════════════════════════════════════════

def test_model_features_exist_in_the_dataset(dataset):
    for feature in FEATURE_MAPPING:
        column = FEATURE_MAPPING[feature]["column"]
        assert column in dataset.columns, f"{column} is missing from the dataset"


def test_base_model_features_are_the_plugin_required_fields():
    """The model and the plugin must accept exactly the same cases."""
    assert set(BASE_FEATURES) == set(CardiovascularPlugin.required_fields)


def test_the_optional_features_are_optional_for_the_plugin():
    assert set(OPTIONAL_FEATURES) <= set(CardiovascularPlugin.optional_fields)


def test_the_extended_variant_adds_exactly_the_optional_features():
    assert VARIANTS["base"] == BASE_FEATURES
    assert VARIANTS["extended"] == BASE_FEATURES + OPTIONAL_FEATURES


def test_every_dataset_column_is_accounted_for(dataset):
    """No column is silently dropped: each is mapped, unavailable, or dataset-only."""
    accounted = {details["column"] for details in FEATURE_MAPPING.values()}
    accounted |= set(DATASET_ONLY_COLUMNS)
    accounted |= {TARGET_COLUMN}
    for column in dataset.columns:
        if column == "patientid":
            continue
        assert column in accounted, f"{column} is not accounted for"


def test_the_dataset_has_no_smoking_column(dataset):
    """The reason smoker is excluded from the model, not an accident."""
    assert not any("smok" in column.lower() for column in dataset.columns)


def test_smoker_is_documented_as_unavailable():
    assert "smoker" in UNAVAILABLE_FEATURES
    assert "smoking column" in UNAVAILABLE_FEATURES["smoker"]
    assert "smoker" not in BASE_FEATURES
    assert "smoker" not in OPTIONAL_FEATURES


def test_fasting_glucose_flag_is_not_mapped_to_glucose():
    """A binary flag is not a concentration; mapping it would compare different units."""
    assert "fastingbloodsugar" in DATASET_ONLY_COLUMNS
    assert "glucose" in UNAVAILABLE_FEATURES


# ═════════════════════════════════════════════════════════════════════════════
#  NOTHING IS INVENTED
# ═════════════════════════════════════════════════════════════════════════════

def test_the_model_ignores_smoking_status():
    """Smoking status cannot change a model score, because the model never saw it."""
    model = CardioModel()
    case = {"age": 58, "systolic_bp": 150, "total_cholesterol": 220}
    without = model.predict_probability(case)
    as_non_smoker = model.predict_probability({**case, "smoker": False})
    as_smoker = model.predict_probability({**case, "smoker": True})

    assert without is not None
    assert without == as_non_smoker == as_smoker


def test_a_missing_required_feature_makes_the_model_decline():
    """The model refuses to score rather than imputing a value."""
    model = CardioModel()
    assert model.predict_probability({"age": 58, "systolic_bp": 150}) is None
    assert model.predict_probability({"age": 58, "systolic_bp": 150, "total_cholesterol": None}) is None
    assert model.predict_probability({"age": "not a number", "systolic_bp": 150, "total_cholesterol": 220}) is None


def test_missing_features_are_reported():
    model = CardioModel()
    assert model.missing_features({"age": 58}) == ["systolic_bp", "total_cholesterol"]
    assert model.missing_features({"age": 58, "systolic_bp": 150, "total_cholesterol": 220}) == []


# ═════════════════════════════════════════════════════════════════════════════
#  UNAVAILABLE IS NOT NEGATIVE
# ═════════════════════════════════════════════════════════════════════════════

def test_unknown_smoking_status_scores_differently_from_a_confirmed_non_smoker():
    """An absent smoking status must not be read as 'does not smoke'."""
    plugin = CardiovascularPlugin()
    # Mid-range values, so the three states give three different scores
    # (with every component saturated all three collapse to the maximum).
    case = {"age": 50, "systolic_bp": 130, "total_cholesterol": 210}

    unknown = plugin.rule_risk(case)
    non_smoker = plugin.rule_risk({**case, "smoker": False})
    smoker = plugin.rule_risk({**case, "smoker": True})

    assert unknown["smoker_status"] == "unknown"
    assert non_smoker["smoker_status"] == "non_smoker"
    assert smoker["smoker_status"] == "current"

    # A confirmed non-smoker genuinely lowers the score; an unknown status
    # neither lowers nor raises it, so it lands between the two.
    assert non_smoker["score"] < unknown["score"] < smoker["score"]


def test_unknown_smoking_status_is_reported_as_an_assumption():
    plugin = CardiovascularPlugin()
    result = plugin.rule_risk({"age": 62, "systolic_bp": 168, "total_cholesterol": 265})

    assert result["assumptions"]
    assert any("Smoking status" in note for note in result["assumptions"])

    factors = {f["parameter"]: f for f in result["risk_factors"]}
    assert factors["smoker"]["impact"] == "Unknown"
    assert factors["smoker"]["value"] is None


def test_unknown_smoking_status_is_flagged():
    plugin = CardiovascularPlugin()
    flags = plugin.flags({"age": 62, "systolic_bp": 168, "total_cholesterol": 265}, {})
    assert "SMOKING STATUS UNKNOWN" in flags
    assert "SMOKING RISK FACTOR" not in flags


def test_a_confirmed_non_smoker_is_not_flagged_as_unknown():
    plugin = CardiovascularPlugin()
    flags = plugin.flags(
        {"age": 62, "systolic_bp": 168, "total_cholesterol": 265, "smoker": False}, {}
    )
    assert "SMOKING STATUS UNKNOWN" not in flags
    assert "SMOKING RISK FACTOR" not in flags


def test_a_cardiovascular_case_needs_no_smoking_status():
    """The plugin applies without a smoking status, and says it scored on partial data."""
    plugin = CardiovascularPlugin()
    assert "smoker" not in plugin.required_fields
    result = plugin.rule_risk({"age": 45, "systolic_bp": 120, "total_cholesterol": 190})
    assert result["status"] == "completed"
    assert result["smoker_status"] == "unknown"


# ═════════════════════════════════════════════════════════════════════════════
#  EVALUATED ON HELD-OUT DATA
# ═════════════════════════════════════════════════════════════════════════════

def test_the_model_was_trained_on_the_real_dataset(metadata):
    assert metadata["training_rows"] == 1000
    assert metadata["data_source"].endswith("Cardiovascular_Disease_Dataset.csv")


def test_the_metadata_reports_held_out_metrics(metadata):
    test = metadata["metrics"]["base"]["test"]
    assert test["rows"] == 200
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "brier"):
        assert metric in test
    # Real but modest discrimination — clearly better than chance.
    assert 0.65 <= test["roc_auc"] <= 0.95
    assert test["brier"] < 0.25


def test_the_model_beats_the_always_positive_baseline(metadata):
    baseline = metadata["metrics"]["base"]["baseline"]
    assert (
        metadata["metrics"]["base"]["test"]["accuracy"]
        > baseline["accuracy_of_always_positive"] + 0.1
    )
    assert baseline["prevalence"] == pytest.approx(0.58, abs=0.01)


def test_cross_validation_is_reported(metadata):
    cv = metadata["metrics"]["base"]["cross_validation"]
    assert cv["folds"] == 5
    assert 0.6 <= cv["roc_auc"] <= 0.95


def test_the_extended_variant_is_measured_too(metadata):
    """The stronger model is held to the same standard as the base one."""
    extended = metadata["metrics"]["extended"]
    assert extended["features"] == BASE_FEATURES + OPTIONAL_FEATURES
    assert extended["test"]["rows"] == 200
    # The two category columns carry a lot of the label's signal.
    assert extended["test"]["roc_auc"] > metadata["metrics"]["base"]["test"]["roc_auc"] + 0.1
    assert extended["cross_validation"]["roc_auc"] > 0.9


def test_confidence_is_capped_because_the_labels_are_rule_derived(metadata):
    assert metadata["confidence"] in ("low", "moderate")
    assert metadata["confidence"] != "high"
    assert "rule-derived" in metadata["confidence_basis"].lower()
    # The stronger variant is capped for the same reason.
    assert metadata["metrics"]["extended"]["confidence"] != "high"
    assert "rule-derived" in metadata["metrics"]["extended"]["confidence_basis"].lower()


def test_the_target_semantics_are_stated(metadata):
    """The label is a disease likelihood, not a 10-year risk."""
    assert "not a 10-year" in metadata["target_semantics"].lower()
    assert metadata["label"] == "target"


def test_the_limitations_are_recorded(metadata):
    limitations = " ".join(metadata["limitations"]).lower()
    assert "rule-derived" in limitations
    assert "smoking status is absent" in limitations
    assert "licence" in limitations


# ═════════════════════════════════════════════════════════════════════════════
#  THE EXTENDED MODEL IS USED WHOLE OR NOT AT ALL
# ═════════════════════════════════════════════════════════════════════════════

BASE_CASE = {"age": 58, "systolic_bp": 150, "total_cholesterol": 220}


def test_the_extended_model_is_used_when_both_optional_features_are_present():
    model = CardioModel()
    assert model.select_variant({**BASE_CASE, "chest_pain_type": 2, "st_slope": 3}) == "extended"


def test_the_base_model_is_used_when_an_optional_feature_is_missing():
    """A partially-supplied case is never scored by a model that wants both."""
    model = CardioModel()
    assert model.select_variant(BASE_CASE) == "base"
    assert model.select_variant({**BASE_CASE, "chest_pain_type": 2}) == "base"
    assert model.select_variant({**BASE_CASE, "st_slope": 3}) == "base"


def test_the_prediction_reports_which_variant_scored_the_case():
    model = CardioModel()

    base = model.predict(BASE_CASE)
    assert base["variant"] == "base"
    assert set(base["missing_optional_features"]) == set(OPTIONAL_FEATURES)
    assert base["features"] == BASE_FEATURES

    extended = model.predict({**BASE_CASE, "chest_pain_type": 2, "st_slope": 3})
    assert extended["variant"] == "extended"
    assert extended["missing_optional_features"] == []
    assert extended["features"] == BASE_FEATURES + OPTIONAL_FEATURES


def test_the_extended_model_scores_differently_from_the_base_model():
    model = CardioModel()
    base = model.predict(BASE_CASE)["probability"]
    extended = model.predict({**BASE_CASE, "chest_pain_type": 2, "st_slope": 3})["probability"]
    assert base != extended


def test_the_plugin_reports_the_variant_and_the_missing_features():
    plugin = CardiovascularPlugin()

    base = plugin.predict_risk(BASE_CASE)
    assert base["model_variant"] == "base"
    assert set(base["model_missing_optional_features"]) == set(OPTIONAL_FEATURES)

    extended = plugin.predict_risk({**BASE_CASE, "chest_pain_type": 2, "st_slope": 3})
    assert extended["model_variant"] == "extended"
    assert extended["model_missing_optional_features"] == []
    assert extended["model_features"] == BASE_FEATURES + OPTIONAL_FEATURES


def test_the_category_codes_are_reported_but_not_ranged():
    """An ordinal code is not a measurement, so it gets no Normal/High status."""
    from Agents.reportanalyzer import validate_parameters

    analysis = validate_parameters(
        {"age": 58, "systolic_bp": 150, "total_cholesterol": 220,
         "chest_pain_type": 2, "st_slope": 3},
        plugin=CardiovascularPlugin(),
    )
    parameters = analysis["parameters"]
    assert parameters["chest_pain_type"]["status"] == "Reported"
    assert parameters["st_slope"]["status"] == "Reported"
    assert parameters["systolic_bp"]["status"] in ("Normal", "High", "Low")


def test_the_category_codes_appear_as_risk_factors():
    plugin = CardiovascularPlugin()
    result = plugin.predict_risk({**BASE_CASE, "chest_pain_type": 2, "st_slope": 3})
    factors = {f["parameter"]: f for f in result["risk_factors"]}
    assert factors["chest_pain_type"]["impact"] == "Reported"
    assert factors["chest_pain_type"]["label"]
    assert factors["st_slope"]["label"]


# ═════════════════════════════════════════════════════════════════════════════
#  THE PREDICTION CARRIES ITS OWN CAVEATS
# ═════════════════════════════════════════════════════════════════════════════

def test_the_prediction_reports_confidence_and_the_target_semantics():
    plugin = CardiovascularPlugin()
    result = plugin.predict_risk({"age": 62, "systolic_bp": 168, "total_cholesterol": 265})

    assert result["prediction_method"] == "ML-based (Logistic Regression)"
    assert result["model_confidence"] in ("low", "moderate")
    assert "not a 10-year" in result["model_target"].lower()
    assert "not a 10-year" in result["disclaimer"].lower()
    assert "not a 10-year" in result["summary"].lower()


def test_the_confidence_reaches_the_unified_response():
    """A caller must be able to see how far the score should be trusted."""
    from Agents.contracts import build_unified_response

    plugin = CardiovascularPlugin()
    risk = plugin.predict_risk({**BASE_CASE, "chest_pain_type": 2, "st_slope": 3})
    unified = build_unified_response(
        {"disease_results": {"cardiovascular": {"risk": risk}}, "workflow_status": "completed"},
        1,
        "2026-01-01",
    )
    assessment = unified["assessments"]["cardiovascular"]
    assert assessment["model_confidence"] == "moderate"
    assert assessment["model_variant"] == "extended"
    assert assessment["risk"]["model_target"]
    assert assessment["risk"]["model_limitations"]


def test_a_case_the_model_cannot_score_falls_back_to_the_rules():
    plugin = CardiovascularPlugin()
    result = plugin.predict_risk({"age": 62})

    assert result["prediction_method"] == "rule_based"
    assert result["fallback_reason"]


def test_the_model_uses_a_single_artifact():
    """One artifact holds the whole pipeline, so a stale scaler cannot linger."""
    from Agents.plugins import cardio_model

    assert cardio_model.DEFAULT_MODEL_PATH.exists()
    # The stub's separate scaler/medians artifacts are gone from the module:
    # the fitted pipeline carries its own imputer and scaler.
    assert not hasattr(cardio_model, "DEFAULT_SCALER_PATH")
    assert not hasattr(cardio_model, "DEFAULT_MEDIANS_PATH")
