"""Evaluate the cardiovascular model against the rule scorer and a baseline.

Run from the repository root::

    python eval/compare.py

The evaluation is honest about what it measures:

* The dataset's labels are rule-derived rather than observed clinical
  outcomes (see ``data/README.md``), so these numbers describe how well the
  model recovers the generating rule — they do not establish clinical
  validity.
* The target is cross-sectional, so the model's output is a disease
  likelihood, not a 10-year risk.
* The model's feature set is fixed to the fields the API accepts, so the
  comparison below is against the rule scorer, which uses the same three
  fields plus the smoking status the dataset does not carry.

Results are written to ``eval/results/``.
"""

import json
import sys
from pathlib import Path

# Allow `python eval/compare.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from Agents.plugins.cardio_model import (
    BASE_FEATURES,
    FEATURE_MAPPING,
    OPTIONAL_FEATURES,
    VARIANTS,
    CardioModel,
    _build_pipeline,
)
from Agents.plugins.cardiovascular import CardiovascularPlugin

SEED = 42
DATA_PATH = Path("data/Cardiovascular_Disease_Dataset.csv")
RESULTS_DIR = Path("eval/results")


def load_dataset():
    """Load the dataset and rename the mapped columns onto API field names."""
    df = pd.read_csv(DATA_PATH)
    rename = {details["column"]: feature for feature, details in FEATURE_MAPPING.items()}
    df = df.rename(columns=rename)
    return df


def score_rows(probabilities, y_true, threshold=0.5):
    """Standard binary metrics for a vector of probabilities."""
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 4),
        "brier": round(float(brier_score_loss(y_true, probabilities)), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def run_evaluation():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    y = df["target"].values

    # ── 1. The shipped models, evaluated on a held-out split ─────────────
    base_features = VARIANTS["base"]
    X = df[base_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    model = CardioModel()
    # Train the shipped artifact on the full dataset. Its own held-out
    # metrics (recorded in the metadata) are the authoritative ones.
    model.train(data_path=DATA_PATH)
    metadata = model.metadata()
    ml_metrics = metadata.get("metrics", {})

    # For an apples-to-apples comparison on the *same* rows, score the test
    # split with a pipeline fitted on the training split only.
    fresh = _build_pipeline().fit(X_train, y_train)
    ml_probabilities = fresh.predict_proba(X_test)[:, 1]
    ml_on_test = score_rows(ml_probabilities, y_test)

    # ── 2. The rule scorer on the same rows ─────────────────────────────
    plugin = CardiovascularPlugin()
    rule_probabilities = []
    for _, row in X_test.iterrows():
        case = {feature: row[feature] for feature in base_features}
        # The dataset has no smoking column, so the rule scorer is evaluated
        # with the smoking status unknown — it drops that component rather
        # than assuming "does not smoke".
        rule_probabilities.append(plugin.rule_risk(case)["risk_probability"])
    rule_on_test = score_rows(rule_probabilities, y_test)

    # ── 3. Baselines ────────────────────────────────────────────────────
    prevalence = float(df["target"].mean())
    always_positive = score_rows(np.full(len(y_test), prevalence), y_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(_build_pipeline(), X, y, cv=cv, method="predict_proba")[:, 1]
    cv_metrics = {
        "roc_auc": round(float(roc_auc_score(y, oof)), 4),
        "accuracy": round(float(accuracy_score(y, (oof >= 0.5).astype(int))), 4),
        "brier": round(float(brier_score_loss(y, oof)), 4),
    }

    # The extended model, on the same rows.
    extended_features = VARIANTS["extended"]
    X_ext = df[extended_features]
    X_ext_train, X_ext_test, y_ext_train, y_ext_test = train_test_split(
        X_ext, y, test_size=0.2, random_state=SEED, stratify=y
    )
    fresh_ext = _build_pipeline().fit(X_ext_train, y_ext_train)
    extended_on_test = score_rows(fresh_ext.predict_proba(X_ext_test)[:, 1], y_ext_test)

    summary = [
        {"config": "ML model, extended features", **extended_on_test},
        {"config": "ML model, base features", **ml_on_test},
        {"config": "Rule scorer (smoking status unknown)", **rule_on_test},
        {"config": "Always-positive baseline", **always_positive},
    ]
    summary_df = pd.DataFrame(summary).drop(columns=["confusion_matrix"])
    pd.set_option("display.width", 200)
    print("\n--- Cardiovascular evaluation (held-out 20% of 1000 rows) ---")
    print(summary_df.to_string(index=False))
    print(f"\n5-fold cross-validated ROC-AUC: {cv_metrics['roc_auc']}")
    print(f"Prevalence: {prevalence:.4f} (an always-positive classifier scores "
          f"{always_positive['accuracy']:.4f} accuracy)")
    print(f"Model confidence: {metadata.get('confidence')} — {metadata.get('confidence_basis')}")

    print("\nLimitations recorded with the model:")
    for limitation in metadata.get("limitations", []):
        print(f"  - {limitation}")

    summary_df.to_csv(RESULTS_DIR / "evaluation_summary.csv", index=False)
    (RESULTS_DIR / "evaluation_detail.json").write_text(
        json.dumps(
            {
                "data_source": str(DATA_PATH),
                "held_out_base": ml_on_test,
                "held_out_extended": extended_on_test,
                "rule_scorer": rule_on_test,
                "always_positive": always_positive,
                "cross_validation": cv_metrics,
                "shipped_model_metrics": ml_metrics,
                "base_features": base_features,
                "extended_features": extended_features,
                "model_confidence": metadata.get("confidence"),
                "confidence_basis": metadata.get("confidence_basis"),
                "limitations": metadata.get("limitations", []),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_df


if __name__ == "__main__":
    run_evaluation()
