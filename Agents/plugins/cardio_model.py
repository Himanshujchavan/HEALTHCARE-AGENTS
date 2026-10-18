from pathlib import Path
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "cardio_model.pkl"
DEFAULT_SCALER_PATH = BASE_DIR / "cardio_scaler.pkl"
DEFAULT_MEDIANS_PATH = BASE_DIR / "cardio_medians.pkl"

_cached_model = None
_cached_scaler = None
_cached_medians = None

class CardioModel:
    """Handles training and prediction for Cardiovascular Risk using Framingham data."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        scaler_path: Path | str = DEFAULT_SCALER_PATH,
        medians_path: Path | str = DEFAULT_MEDIANS_PATH,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.medians_path = Path(medians_path)

    def train(self, data_path: Path | str = "data/framingham.csv"):
        logger.info(f"Training Cardiovascular ML model using {data_path}...")
        df = pd.read_csv(data_path)

        features = ["age", "systolic_bp", "total_cholesterol", "smoker"]
        X = df[features]
        y = df["TenYearCHD"]

        medians = X.median().to_dict()
        X = X.fillna(medians)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        joblib.dump(model, self.model_path)
        joblib.dump(scaler, self.scaler_path)
        joblib.dump(medians, self.medians_path)

        global _cached_model, _cached_scaler, _cached_medians
        _cached_model = model
        _cached_scaler = scaler
        _cached_medians = medians

        logger.info(f"Model saved to {self.model_path}")
        return accuracy_score(y_test, model.predict(X_test))

    def _ensure_loaded(self):
        global _cached_model, _cached_scaler, _cached_medians
        if _cached_model is None and self.model_path.exists():
            _cached_model = joblib.load(self.model_path)
        if _cached_scaler is None and self.scaler_path.exists():
            _cached_scaler = joblib.load(self.scaler_path)
        if _cached_medians is None and self.medians_path.exists():
            _cached_medians = joblib.load(self.medians_path)

    def predict_probability(self, data: dict) -> float | None:
        """Predicts the probability of heart disease for a given set of parameters."""
        try:
            self._ensure_loaded()
            global _cached_model, _cached_scaler, _cached_medians
            if _cached_model is None or _cached_scaler is None:
                # Try to train if model file does not exist
                self.train()

            features = ["age", "systolic_bp", "total_cholesterol", "smoker"]
            medians = _cached_medians or {"age": 50.0, "systolic_bp": 130.0, "total_cholesterol": 210.0, "smoker": 0.0}

            input_vals = []
            for f in features:
                val = data.get(f)
                if val is None:
                    val = medians.get(f, 0.0)
                else:
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        val = medians.get(f, 0.0)
                input_vals.append(val)

            X_scaled = _cached_scaler.transform([input_vals])
            prob = _cached_model.predict_proba(X_scaled)[0][1]
            return float(prob)
        except Exception as e:
            logger.warning(f"ML Prediction Error: {e}")
            return None

if __name__ == "__main__":
    cm = CardioModel()
    acc = cm.train()
    print(f"Training complete. Test Accuracy: {acc:.2f}")
