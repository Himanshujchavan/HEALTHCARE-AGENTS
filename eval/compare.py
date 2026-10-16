import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from Agents.plugins.cardiovascular import CardiovascularPlugin
from Agents.plugins.diabetes import DiabetesPlugin
from Agents.plugins.registry import register

# Initialize plugins for evaluation
register(DiabetesPlugin())
register(CardiovascularPlugin())

def evaluate_config(name, data, target, prediction_func):
    """Evaluates a specific risk prediction configuration."""
    y_true = target
    y_pred_probs = [prediction_func(row) for _, row in data.iterrows()]

    # Threshold at 0.5 for binary classification
    y_pred = [1 if p >= 0.5 else 0 for p in y_pred_probs]

    return {
        "config": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred_probs),
        "cm": confusion_matrix(y_true, y_pred)
    }

def run_evaluation():
    # Load Framingham data
    df = pd.read_csv("data/framingham.csv")
    features = ["age", "systolic_bp", "total_cholesterol", "smoker"]
    X = df[features]
    y = df["TenYearCHD"]

    # 1. Rule-Only Implementation
    def rule_only(row):
        # Mock rule-based logic (simplified)
        score = 0
        if row['systolic_bp'] > 140: score += 0.4
        if row['total_cholesterol'] > 240: score += 0.4
        if row['smoker'] == 1: score += 0.2
        return score

    # 2. ML-Only Implementation
    from Agents.plugins.cardio_model import CardioModel
    model = CardioModel()
    def ml_only(row):
        return model.predict_probability(row.to_dict())

    # 3. Hybrid Implementation (Our system)
    def hybrid_only(row):
        # Hybrid uses ML but checks rule-based flags for emergency boosters
        prob = ml_only(row)
        if row['systolic_bp'] > 160: prob = max(prob, 0.8)
        return prob

    # 4. LLM-Only (Mocked for evaluation purposes)
    def llm_only(row):
        return np.random.uniform(0, 1)

    configs = [
        ("Rule-Only", rule_only),
        ("ML-Only", ml_only),
        ("Hybrid", hybrid_only),
        ("LLM-Only", llm_only)
    ]

    results = []
    for name, func in configs:
        res = evaluate_config(name, X, y, func)
        results.append(res)

        # Plot Confusion Matrix
        fig, ax = plt.subplots(figsize=(4,4))
        ConfusionMatrixDisplay(confusion_matrix=res["cm"]).plot(ax=ax)
        plt.title(f"CM: {name}")
        plt.savefig(f"eval/results/cm_{name.lower().replace('-', '_')}.png")
        plt.close()

    # Print Summary Table
    summary_df = pd.DataFrame(results).drop(columns=["cm"])
    print("\n--- Cardiovascular Risk Evaluation Summary ---")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("eval/results/evaluation_summary.csv", index=False)

if __name__ == "__main__":
    run_evaluation()
