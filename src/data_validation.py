"""Great Expectations checks for prediction-time data."""

import pandas as pd

from src.validation import REQUIRED_COLUMNS
from src.config import config


def validate_with_great_expectations(df: pd.DataFrame) -> None:
    """Run GE expectations, with the local validator as a dependency-light fallback."""
    if not config["validation"].get("great_expectations_enabled", False):
        from src.validation import validate_orders

        validate_orders(df)
        return
    try:
        import great_expectations as gx
    except ImportError:
        from src.validation import validate_orders

        validate_orders(df)
        return

    dataset = gx.from_pandas(df)
    for column in REQUIRED_COLUMNS:
        result = dataset.expect_column_to_exist(column)
        if not result["success"]:
            raise ValueError(f"Great Expectations: missing column {column}")
    result = dataset.expect_column_values_to_not_be_null("customer_state")
    if not result["success"]:
        raise ValueError("Great Expectations: customer_state contains null values")
    result = dataset.expect_column_values_to_be_between(
        "customer_zip_code_prefix", min_value=0, max_value=99999
    )
    if not result["success"]:
        raise ValueError("Great Expectations: ZIP code is outside the valid range")