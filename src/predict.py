import joblib
import pandas as pd
from src.config import config

# Load the trained model (Never re-train in inference)
model = joblib.load(config['paths']['model_path'])
MODEL_VERSION = config["model"]["version"]
DECISION_THRESHOLD = config["model"]["decision_threshold"]


def _prediction_result(prediction, probability) -> dict:
    return {
        "prediction": int(prediction),
        "is_late": bool(prediction == 1),
        "probability": round(float(probability), 15),
        "model_version": MODEL_VERSION,
    }


def model_info() -> dict:
    return {
        "model_name": type(model).__name__,
        "model_version": MODEL_VERSION,
        "decision_threshold": DECISION_THRESHOLD,
        "feature_count": len(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None,
    }

def make_prediction(processed_df: pd.DataFrame) -> dict:
    """
    Predicts whether an order will be late.
    Returns the prediction, probability, and model version.
    """
    # Get the prediction (0 for on-time, 1 for late)
    probabilities = model.predict_proba(processed_df)[:, 1]
    prediction = int(probabilities[0] >= DECISION_THRESHOLD)
    
    # Get the probability of the late class (class 1)
    probability = probabilities[0]
    
    return _prediction_result(prediction, probability)


def make_batch_prediction(processed_df: pd.DataFrame) -> list[dict]:
    probabilities = model.predict_proba(processed_df)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    return [_prediction_result(prediction, probability) for prediction, probability in zip(predictions, probabilities)]