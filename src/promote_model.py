"""Offline model promotion from the Task 2 artifacts into production models/."""

import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PROJECT_ROOT
from src.tracking import track_promotion

RAW_COLUMNS = [
    "customer_state",
    "customer_zip_code_prefix",
    "n_items",
    "total_price",
    "total_freight",
    "total_payment_value",
    "n_payments",
    "max_installments",
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
]
NUMERIC_COLUMNS = [
    "n_items",
    "total_price",
    "total_freight",
    "total_payment_value",
    "n_payments",
    "max_installments",
]
SCALE_COLUMNS = [
    "customer_zip_code_prefix",
    "n_items",
    "total_price",
    "total_freight",
    "total_payment_value",
    "n_payments",
    "max_installments",
    "order_month",
    "order_weekday",
    "estimated_days",
    "order_year",
    "order_month_sin",
    "order_month_cos",
    "order_weekday_sin",
    "order_weekday_cos",
]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[RAW_COLUMNS].copy()
    purchase = pd.to_datetime(data.pop("order_purchase_timestamp"), errors="coerce")
    estimated = pd.to_datetime(data.pop("order_estimated_delivery_date"), errors="coerce")
    data["order_month"] = purchase.dt.month
    data["order_weekday"] = purchase.dt.weekday
    data["estimated_days"] = (estimated - purchase).dt.total_seconds() / 86400
    data["order_year"] = purchase.dt.year - 2016
    month_angle = 2 * math.pi * data["order_month"] / 12
    weekday_angle = 2 * math.pi * data["order_weekday"] / 7
    data["order_month_sin"] = month_angle.map(math.sin)
    data["order_month_cos"] = month_angle.map(math.cos)
    data["order_weekday_sin"] = weekday_angle.map(math.sin)
    data["order_weekday_cos"] = weekday_angle.map(math.cos)
    return data.reset_index(drop=True)


def select_threshold(y_true, probabilities) -> float:
    thresholds = np.arange(0.10, 0.81, 0.01)
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def promote(artifacts_dir: Path, output_dir: Path) -> dict:
    frames = {
        split: pd.read_csv(
            artifacts_dir / f"{split}.csv",
            parse_dates=["order_purchase_timestamp", "order_estimated_delivery_date"],
        ).dropna(subset=["is_late", "order_purchase_timestamp", "order_estimated_delivery_date"])
        for split in ("train", "val", "test")
    }
    train_features = build_features(frames["train"])
    val_features = build_features(frames["val"])
    test_features = build_features(frames["test"])

    imputer = SimpleImputer(strategy="median")
    train_features[NUMERIC_COLUMNS] = imputer.fit_transform(train_features[NUMERIC_COLUMNS])
    for features in (val_features, test_features):
        features[NUMERIC_COLUMNS] = imputer.transform(features[NUMERIC_COLUMNS])

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_encoded = encoder.fit_transform(train_features[["customer_state"]])
    val_encoded = encoder.transform(val_features[["customer_state"]])
    test_encoded = encoder.transform(test_features[["customer_state"]])

    scaler = StandardScaler()
    train_features[SCALE_COLUMNS] = scaler.fit_transform(train_features[SCALE_COLUMNS])
    val_features[SCALE_COLUMNS] = scaler.transform(val_features[SCALE_COLUMNS])
    test_features[SCALE_COLUMNS] = scaler.transform(test_features[SCALE_COLUMNS])

    encoded_columns = encoder.get_feature_names_out(["customer_state"])
    feature_columns = SCALE_COLUMNS + list(encoded_columns)
    x_train = pd.concat([train_features[SCALE_COLUMNS], pd.DataFrame(train_encoded, columns=encoded_columns)], axis=1)
    x_val = pd.concat([val_features[SCALE_COLUMNS], pd.DataFrame(val_encoded, columns=encoded_columns)], axis=1)
    x_test = pd.concat([test_features[SCALE_COLUMNS], pd.DataFrame(test_encoded, columns=encoded_columns)], axis=1)

    model = LogisticRegression(class_weight="balanced", C=0.1, max_iter=2000, random_state=42)
    model.fit(x_train[feature_columns], frames["train"]["is_late"].astype(int))
    val_probability = model.predict_proba(x_val[feature_columns])[:, 1]
    threshold = select_threshold(frames["val"]["is_late"].astype(int), val_probability)
    test_probability = model.predict_proba(x_test[feature_columns])[:, 1]
    test_prediction = test_probability >= threshold
    metrics = {
        "model": type(model).__name__,
        "threshold": threshold,
        "validation_f1": float(f1_score(frames["val"]["is_late"], val_probability >= threshold, zero_division=0)),
        "test_f1": float(f1_score(frames["test"]["is_late"], test_prediction, zero_division=0)),
        "test_recall": float(recall_score(frames["test"]["is_late"], test_prediction, zero_division=0)),
        "test_precision": float(precision_score(frames["test"]["is_late"], test_prediction, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(frames["test"]["is_late"], test_probability)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "final_model.joblib")
    joblib.dump(imputer, output_dir / "num_imputer.joblib")
    joblib.dump(encoder, output_dir / "state_encoder.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    (output_dir / "feature_list.txt").write_text("\n".join(feature_columns) + "\n", encoding="utf-8")
    (output_dir / "promotion_metrics.yaml").write_text(yaml.safe_dump(metrics, sort_keys=False), encoding="utf-8")
    track_promotion(
        metrics,
        model,
        [
            output_dir / "final_model.joblib",
            output_dir / "num_imputer.joblib",
            output_dir / "state_encoder.joblib",
            output_dir / "scaler.joblib",
            output_dir / "feature_list.txt",
            output_dir / "promotion_metrics.yaml",
        ],
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT.parent / "task2_notebooks" / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models")
    args = parser.parse_args()
    metrics = promote(args.artifacts_dir, args.output_dir)
    print(yaml.safe_dump(metrics, sort_keys=False))


if __name__ == "__main__":
    main()
