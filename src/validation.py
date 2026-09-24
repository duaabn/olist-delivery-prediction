"""Validation for inference payloads before they reach fitted transformers."""

import pandas as pd

from src.config import config

REQUIRED_COLUMNS = {
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
}


def validate_orders(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if df.empty:
        raise ValueError("At least one order is required")
    max_batch_size = config["validation"]["max_batch_size"]
    if len(df) > max_batch_size:
        raise ValueError(f"Batch size cannot exceed {max_batch_size}")

    numeric_columns = REQUIRED_COLUMNS - {
        "customer_state",
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
    }
    numeric_values = df[list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    if numeric_values.isna().all(axis=1).any():
        raise ValueError("Each order must contain numeric feature values")
    if (numeric_values < 0).any().any():
        raise ValueError("Numeric order features cannot be negative")

    states = config["validation"].get("allowed_states", [])
    if states and (~df["customer_state"].isin(states)).any():
        raise ValueError("customer_state contains an unsupported category")
