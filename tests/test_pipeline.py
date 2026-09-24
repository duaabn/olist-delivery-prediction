import pandas as pd
import pytest

from src.predict import make_batch_prediction, make_prediction
from src.preprocess import apply_preprocessing
from src.validation import validate_orders


def test_preprocessing_matches_saved_feature_contract(sample_order):
    processed = apply_preprocessing(pd.DataFrame([sample_order]))
    assert processed.shape == (1, 42)
    assert processed.columns[0] == "customer_zip_code_prefix"
    assert processed.columns[-1] == "customer_state_TO"


def test_single_and_batch_predictions_match(sample_order):
    processed = apply_preprocessing(pd.DataFrame([sample_order, sample_order]))
    batch_results = make_batch_prediction(processed)
    single_result = make_prediction(processed.iloc[[0]])
    assert batch_results[0] == single_result
    assert len(batch_results) == 2


def test_unknown_state_is_supported_by_fitted_encoder(sample_order):
    sample_order["customer_state"] = "XX"
    processed = apply_preprocessing(pd.DataFrame([sample_order]))
    assert processed.shape == (1, 42)


def test_invalid_timestamp_is_rejected(sample_order):
    sample_order["order_purchase_timestamp"] = "not-a-date"
    with pytest.raises(ValueError, match="valid timestamps"):
        apply_preprocessing(pd.DataFrame([sample_order]))


def test_negative_values_are_rejected(sample_order):
    sample_order["total_price"] = -1
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_orders(pd.DataFrame([sample_order]))
