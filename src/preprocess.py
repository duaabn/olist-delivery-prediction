import math

import joblib
import pandas as pd
from src.config import config

# Load fitted objects (Never re-fit in inference)
num_imputer = joblib.load(config['paths']['imputer_path'])
state_encoder = joblib.load(config['paths']['encoder_path'])
scaler = joblib.load(config['paths']['scaler_path'])

def build_base_features(df):
    """Extracts date features and selects required columns."""
    df = df.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )
    df["order_estimated_delivery_date"] = pd.to_datetime(
        df["order_estimated_delivery_date"], errors="coerce"
    )
    if df[["order_purchase_timestamp", "order_estimated_delivery_date"]].isna().any().any():
        raise ValueError("order_purchase_timestamp and order_estimated_delivery_date must contain valid timestamps")
    df["order_month"] = df["order_purchase_timestamp"].dt.month
    df["order_weekday"] = df["order_purchase_timestamp"].dt.weekday
    df["estimated_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    df["order_year"] = df["order_purchase_timestamp"].dt.year - 2016
    month_angle = 2 * math.pi * df["order_month"] / 12
    weekday_angle = 2 * math.pi * df["order_weekday"] / 7
    df["order_month_sin"] = month_angle.map(math.sin)
    df["order_month_cos"] = month_angle.map(math.cos)
    df["order_weekday_sin"] = weekday_angle.map(math.sin)
    df["order_weekday_cos"] = weekday_angle.map(math.cos)
    
    feature_cols = [
        "customer_state", "customer_zip_code_prefix",
        "n_items", "total_price", "total_freight",
        "total_payment_value", "n_payments", "max_installments",
        "order_month", "order_weekday", "estimated_days", "order_year",
        "order_month_sin", "order_month_cos", "order_weekday_sin",
        "order_weekday_cos"
    ]
    return df[feature_cols]

def apply_preprocessing(df):
    """Applies imputation, encoding, and scaling exactly as in Notebook 5."""
    df = build_base_features(df)
    
    # 1. Imputation
    num_cols = config['features']['numerical_cols']
    df[num_cols] = num_imputer.transform(df[num_cols])
    
    # 2. Encoding
    cat_cols = config['features']['categorical_cols']
    encoded_arr = state_encoder.transform(df[cat_cols])
    encoded_cols = state_encoder.get_feature_names_out(cat_cols)
    encoded_df = pd.DataFrame(encoded_arr, columns=encoded_cols, index=df.index)
    
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded_df], axis=1)
    
    # 3. Scaling
    scale_cols = config['features']['scale_cols']
    df[scale_cols] = scaler.transform(df[scale_cols])
    
    # 4. Ensure column order matches the trained feature list
    with open(config['paths']['feature_list_path'], "r") as f:
        expected_features = f.read().splitlines()
        
    return df[expected_features]