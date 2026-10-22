"""Cardiovascular ML model.

Logistic regression over `data/Cardiovascular_Disease_Dataset.csv`, fitted on
the subset of dataset columns that map onto fields the API actually accepts.

The rules this module follows:

* **Nothing is invented.** If the training data has no column for a field —
  ``smoker`` is the important one — the field is excluded from the model and
  the exclusion is recorded in the training metadata. A missing feature is
  never silently treated as a negative value.
* **The model schema and the plugin schema are the same set.** ``BASE_FEATURES``
  below is exactly ``CardiovascularPlugin.required_fields``, so a case the
  plugin accepts can always be scored and a case the plugin rejects can never
  be scored.
* **Optional features are used whole or not at all.** The dataset also carries
  two strong features the API accepts optionally (``chest_pain_type`` and
  ``st_slope``). A second, "extended" model is trained on them; it is used
  only when *both* are present, and the base model is used otherwise. A
  partially-supplied case is never scored by a model that expects a feature
  it does not have.
* **A missing input is a refusal, not an imputation.** At inference a missing
  required feature makes the model decline to predict; the plugin then falls
  back to the rule-based scorer, which handles partial data explicitly.
* **The model reports what it measured.** Training evaluates each variant on
  held-out data and on cross-validation, and stores the metrics in the
  metadata file so the plugin can tell the caller how far the probability
  should be trusted.

The dataset's labels are rule-derived rather than observed (see
`data/README.md`), so the reported metrics measure how well the model recovers
the generating rule — not clinical validity. The confidence level is capped
accordingly.
"""

from pathlib import Path
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "cardio_model.pkl"
DEFAULT_METADATA_PATH = BASE_DIR / "cardio_model_metadata.json"
DEFAULT_DATA_PATH = Path("data/Cardiovascular_Disease_Dataset.csv")

#: Features every cardiovascular case has. Deliberately identical to the
#: cardiovascular plugin's required fields.
BASE_FEATURES: List[str] = ["age", "systolic_bp", "total_cholesterol"]

#: Features the API accepts optionally. The extended model uses them when
#: they are all present.
OPTIONAL_FEATURES: List[str] = ["chest_pain_type", "st_slope"]

#: How each model feature is read from the dataset. ``restingBP``,
#: ``serumcholestrol``, ``chestpain`` and ``slope`` keep the upstream
#: dataset's spelling.
FEATURE_MAPPING: Dict[str, Dict[str, str]] = {
    "age": {
        "column": "age",
        "unit": "years",
        "description": "Patient age.",
    },
    "systolic_bp": {
        "column": "restingBP",
        "unit": "mmHg",
        "description": "Resting blood pressure (systolic).",
    },
    "total_cholesterol": {
        "column": "serumcholestrol",
        "unit": "mg/dL",
        "description": "Serum cholesterol.",
    },
    "chest_pain_type": {
        "column": "chestpain",
        "unit": "category (0-3)",
        "description": (
            "Chest pain type, ordered from no characteristic pain to "
            "asymptomatic. Ordinal code from the source dataset."
        ),
    },
    "st_slope": {
        "column": "slope",
        "unit": "category (0-3)",
        "description": (
            "Peak-exercise ST segment slope, ordered from downsloping to "
            "upsloping. Ordinal code from the source dataset."
        ),
    },
}

#: The feature set of each trained variant.
VARIANTS: Dict[str, List[str]] = {
    "base": list(BASE_FEATURES),
    "extended": list(BASE_FEATURES) + list(OPTIONAL_FEATURES),
}

TARGET_COLUMN = "target"

TARGET_SEMANTICS = (
    "target = 1 means the patient is recorded as having cardiovascular disease "
    "in a cross-sectional dataset. It carries no time horizon, so the model's "
    "probability is a disease likelihood and NOT a 10-year or any other "
    "time-to-event risk estimate."
)

#: Fields the API accepts that the training data cannot supply. Recorded so a
#: reader can tell an absent feature apart from a modelled one.
UNAVAILABLE_FEATURES: Dict[str, str] = {
    "smoker": (
        "The dataset has no smoking column. The value is not fabricated and "
        "not assumed to be zero, so the model score is independent of smoking "
        "status."
    ),
    "diastolic_bp": "Not present in the dataset.",
    "ldl": "Not present in the dataset.",
    "hdl": "Not present in the dataset.",
    "bmi": "Not present in the dataset.",
    "glucose": (
        "'fastingbloodsugar' is a binary flag for fasting glucose above 120 "
        "mg/dL, not a glucose concentration, so it is not the same quantity as "
        "the API's 'glucose'."
    ),
}

#: Dataset columns the API does not accept. They carry much of the label's
#: signal, which bounds how well the model can do.
DATASET_ONLY_COLUMNS: Dict[str, str] = {
    "gender": "Sex; not accepted by the API.",
    "restingrelectro": "Resting ECG result; not accepted by the API.",
    "maxheartrate": "Maximum heart rate; not accepted by the API.",
    "exerciseangia": "Exercise-induced angina; not accepted by the API.",
    "oldpeak": "ST depression; not accepted by the API.",
    "noofmajorvessels": "Major vessels coloured by fluoroscopy; not accepted by the API.",
    "fastingbloodsugar": "Binary fasting-glucose flag; not accepted by the API.",
}

#: The label is rule-derived, so no amount of held-out accuracy makes this a
#: clinically validated model. The confidence level is capped accordingly.
MAX_CONFIDENCE = "moderate"

#: Minimum rows before a fitted model is considered beyond the stub range.
MIN_CONFIDENCE_ROW_THRESHOLD = 500

LIMITATIONS: List[str] = [
    "The dataset's labels are rule-derived rather than observed clinical "
    "outcomes, so held-out metrics measure how well the model recovers the "
    "generating rule and do not establish clinical validity.",
    "The target is cross-sectional: the output is a disease likelihood, not a "
    "time-to-event risk estimate, and must not be presented as a 10-year risk.",
    "Smoking status is absent from the training data, so the model cannot "
    "account for it; the rule-based scorer does, and reports an unknown "
    "smoking status differently from a confirmed non-smoker.",
    "The extended model needs both chest pain type and ST slope. A case that "
    "supplies only one of them is scored by the weaker base model, so the "
    "reported confidence does not depend on how the case was filled in.",
    "The dataset still has columns the API does not accept (resting ECG, "
    "maximum heart rate, exercise angina, ST depression, vessels coloured), "
    "which carry part of the label's signal.",
    "The dataset is small relative to clinical risk models and comes from a "
    "single source population, so performance on other populations is unknown.",
    "The dataset licence could not be independently verified for this copy.",
]


def _build_pipeline() -> Pipeline:
    """Impute gaps in the training data, scale, then fit the classifier."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, random_state=42),
            ),
        ]
    )


def _confidence_for(rows: int, cv_auc: float) -> tuple:
    """Confidence level and the reason behind it."""
    if rows < MIN_CONFIDENCE_ROW_THRESHOLD or cv_auc < 0.70:
        level = "low"
        basis = (
            f"Only {rows} training rows and a cross-validated ROC-AUC of "
            f"{cv_auc:.3f}; the probability is indicative only."
        )
    elif cv_auc >= 0.85:
        level = "high"
        basis = (
            f"{rows} training rows and a cross-validated ROC-AUC of {cv_auc:.3f}."
        )
    else:
        level = "moderate"
        basis = (
            f"{rows} training rows and a cross-validated ROC-AUC of {cv_auc:.3f}: "
            "real but modest discrimination."
        )

    if level == "high" and MAX_CONFIDENCE != "high":
        level = MAX_CONFIDENCE
        basis += (
            f" Capped at '{MAX_CONFIDENCE}' because the dataset's labels are "
            "rule-derived, so the metrics cannot establish clinical validity."
        )
    else:
        # The ceiling is the same whatever the metrics say, so say so.
        basis += (
            " The confidence is capped at "
            f"'{MAX_CONFIDENCE}' because the dataset's labels are rule-derived, "
            "so the metrics cannot establish clinical validity."
        )
    return level, basis


class CardioModel:
    """Logistic-regression cardiovascular model over the mapped dataset columns.

    Two variants are trained: ``base`` (the features every case has) and
    ``extended`` (base plus the optional chest pain type and ST slope). A
    prediction uses the extended model only when every optional feature is
    present.
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        metadata_path: Path | str = DEFAULT_METADATA_PATH,
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

    # ── training ──────────────────────────────────────────────────────────

    def train(
        self,
        data_path: Path | str = DEFAULT_DATA_PATH,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Fit both variants, evaluate them on held-out data, and persist them.

        Returns the metrics dictionary that is also written to the metadata
        file.
        """
        data_path = Path(data_path)
        logger.info("Training cardiovascular model from %s", data_path)
        df = pd.read_csv(data_path)

        needed = {details["column"] for details in FEATURE_MAPPING.values()}
        missing = sorted(column for column in needed if column not in df.columns)
        if missing:
            raise ValueError(
                f"{data_path} is missing the column(s) {missing} required to "
                f"train the cardiovascular model."
            )
        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"{data_path} is missing the target column '{TARGET_COLUMN}'."
            )

        y = df[TARGET_COLUMN]
        pipelines: Dict[str, Pipeline] = {}
        metrics: Dict[str, Any] = {}

        for variant, features in VARIANTS.items():
            # Rename the dataset columns onto the API's field names so the
            # fitted pipeline's feature names match what the plugin passes in.
            X = df[[FEATURE_MAPPING[f]["column"] for f in features]].copy()
            X.columns = features

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            pipeline = _build_pipeline()
            pipeline.fit(X_train, y_train)
            pipelines[variant] = pipeline

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            oof = cross_val_predict(
                _build_pipeline(), X, y, cv=cv, method="predict_proba"
            )[:, 1]

            prevalence = float(y.mean())
            confidence, confidence_basis = _confidence_for(
                len(df), float(roc_auc_score(y, oof))
            )

            metrics[variant] = {
                "features": list(features),
                "test": self._evaluate(pipeline, X_test, y_test),
                "cross_validation": {
                    "folds": cv_folds,
                    "roc_auc": round(float(roc_auc_score(y, oof)), 4),
                    "accuracy": round(float(accuracy_score(y, (oof >= 0.5).astype(int))), 4),
                    "average_precision": round(float(average_precision_score(y, oof)), 4),
                    "brier": round(float(brier_score_loss(y, oof)), 4),
                },
                "baseline": {
                    "prevalence": round(prevalence, 4),
                    "accuracy_of_always_positive": round(max(prevalence, 1 - prevalence), 4),
                },
                "confidence": confidence,
                "confidence_basis": confidence_basis,
            }

        joblib.dump(
            {"pipelines": pipelines, "features": {k: list(v) for k, v in VARIANTS.items()}},
            self.model_path,
        )
        self._write_metadata(
            data_path=data_path,
            rows=int(len(df)),
            metrics=metrics,
            random_state=random_state,
        )

        global _cached_artifact, _cached_metadata
        _cached_artifact = {"pipelines": pipelines, "features": {k: list(v) for k, v in VARIANTS.items()}}
        _cached_metadata = self._read_metadata()

        for variant in VARIANTS:
            logger.info(
                "Cardiovascular model [%s]: test AUC %.3f, CV AUC %.3f (%s)",
                variant,
                metrics[variant]["test"]["roc_auc"],
                metrics[variant]["cross_validation"]["roc_auc"],
                metrics[variant]["confidence"],
            )
        return metrics

    def _evaluate(self, pipeline: Pipeline, X_test, y_test) -> Dict[str, Any]:
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        return {
            "rows": int(len(y_test)),
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "average_precision": round(float(average_precision_score(y_test, probabilities)), 4),
            "brier": round(float(brier_score_loss(y_test, probabilities)), 4),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        }

    def _write_metadata(
        self,
        *,
        data_path: Path,
        rows: int,
        metrics: Dict[str, Any],
        random_state: int,
    ):
        metadata = {
            "data_source": str(data_path).replace("\\", "/"),
            "data_documentation": "data/README.md",
            "training_rows": rows,
            "variants": list(VARIANTS),
            "base_features": list(BASE_FEATURES),
            "optional_features": list(OPTIONAL_FEATURES),
            "feature_mapping": {
                feature: dict(details) for feature, details in FEATURE_MAPPING.items()
            },
            "unavailable_features": dict(UNAVAILABLE_FEATURES),
            "dataset_only_columns": dict(DATASET_ONLY_COLUMNS),
            "label": TARGET_COLUMN,
            "target_semantics": TARGET_SEMANTICS,
            "metrics": metrics,
            "random_state": random_state,
            "confidence": metrics["base"]["confidence"],
            "confidence_basis": metrics["base"]["confidence_basis"],
            "limitations": list(LIMITATIONS),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "environment": {
                "python": platform.python_version(),
                "scikit_learn": _sklearn_version(),
            },
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    def _read_metadata(self) -> Dict[str, Any]:
        if not self.metadata_path.exists():
            return {}
        try:
            return json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    # ── inference ─────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        global _cached_artifact, _cached_metadata
        if _cached_artifact is None and self.model_path.exists():
            try:
                _cached_artifact = joblib.load(self.model_path)
            except Exception as exc:  # pragma: no cover - corrupted artifact
                logger.warning("Could not load cardiovascular model: %s", exc)
                _cached_artifact = None
        if _cached_metadata is None:
            _cached_metadata = self._read_metadata()

    def metadata(self) -> Dict[str, Any]:
        """Training metadata: metrics, feature mapping, confidence, limitations."""
        self._ensure_loaded()
        return dict(_cached_metadata or {})

    def missing_features(self, data: Dict[str, Any]) -> List[str]:
        """The base features that are absent from ``data``."""
        return [f for f in BASE_FEATURES if not _is_number(data.get(f))]

    def missing_optional_features(self, data: Dict[str, Any]) -> List[str]:
        """The optional features that are absent from ``data``."""
        return [f for f in OPTIONAL_FEATURES if not _is_number(data.get(f))]

    def select_variant(self, data: Dict[str, Any]) -> str:
        """The variant that will score this case.

        The extended model needs every optional feature; a case missing any of
        them is scored by the base model.
        """
        if not self.missing_optional_features(data):
            return "extended"
        return "base"

    def predict(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Score a case, reporting which variant was used and why.

        Returns ``None`` when the case cannot be scored — either because the
        model artifact is missing or because a required feature is absent. A
        missing feature is never imputed and never read as a negative value.
        """
        try:
            self._ensure_loaded()
            global _cached_artifact
            if not _cached_artifact:
                logger.warning(
                    "Cardiovascular model artifact is missing; declining to predict."
                )
                return None

            missing = self.missing_features(data)
            if missing:
                logger.info(
                    "Declining cardiovascular prediction: missing feature(s) %s",
                    ", ".join(missing),
                )
                return None

            variant = self.select_variant(data)
            features = _cached_artifact["features"][variant]
            pipeline = _cached_artifact["pipelines"][variant]

            values = {feature: float(data[feature]) for feature in features}
            frame = pd.DataFrame([values], columns=features)
            probability = float(pipeline.predict_proba(frame)[0][1])

            return {
                "probability": probability,
                "variant": variant,
                "features": list(features),
                "missing_optional_features": self.missing_optional_features(data),
            }
        except Exception as exc:
            logger.warning("Cardiovascular ML prediction error: %s", exc)
            return None

    def predict_probability(self, data: Dict[str, Any]) -> Optional[float]:
        """Estimated probability of cardiovascular disease for one case."""
        result = self.predict(data)
        return None if result is None else result["probability"]


_cached_artifact = None
_cached_metadata = None


def _is_number(value: Any) -> bool:
    """True for a usable numeric value. Booleans and blanks are not."""
    if value is None or isinstance(value, bool):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _sklearn_version() -> str:
    try:
        import sklearn

        return sklearn.__version__
    except Exception:  # pragma: no cover
        return sys.version.split()[0]


if __name__ == "__main__":
    model = CardioModel()
    results = model.train()
    print(json.dumps(results, indent=2))
