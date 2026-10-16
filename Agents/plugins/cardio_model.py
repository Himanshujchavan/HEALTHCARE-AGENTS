import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class CardioModel:
    """Handles training and prediction for Cardiovascular Risk using Framingham data."""

    def __init__(self, model_path="Agents/plugins/cardio_model.pkl", scaler_path="Agents/plugins/cardio_scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None

    def train(self, data_path="data/framingham.csv"):
        print(f"Training Cardiovascular ML model using {data_path}...")
        df = pd.read_csv(data_path)

        # Features and Target
        features = ["age", "systolic_bp", "total_cholesterol", "smoker"]
        X = df[features]
        y = df["TenYearCHD"]

        # Median Imputation
        X = X.fillna(X.median())

        # Scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Model
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)

        # Save
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

        print(f"Model saved to {self.model_path}")
        return accuracy_score(y_test, self.model.predict(X_test))

    def predict_probability(self, data: dict) -> float:
        """Predicts the probability of heart disease for a given set of parameters."""
        try:
            if self.model is None:
                self.model = joblib.load(self.model_path)
            if self.scaler is None:
                self.scaler = joblib.load(self.scaler_path)

            features = ["age", "systolic_bp", "total_cholesterol", "smoker"]
            # Ensure data is in correct order and handle missing values
            input_vals = [data.get(f, 0) for f in features]
            X_scaled = self.scaler.transform([input_vals])

            # Probability of class 1 (CHD)
            prob = self.model.predict_proba(X_scaled)[0][1]
            return float(prob)
        except Exception as e:
            print(f"ML Prediction Error: {e}")
            return None

if __name__ == "__main__":
    # Self-train on startup for demo purposes
    cm = CardioModel()
    acc = cm.train()
    print(f"Training complete. Test Accuracy: {acc:.2f}")
