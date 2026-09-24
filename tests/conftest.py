import pytest


@pytest.fixture
def sample_order():
    return {
        "customer_state": "SP",
        "customer_zip_code_prefix": 14600,
        "n_items": 1.0,
        "total_price": 29.9,
        "total_freight": 15.56,
        "total_payment_value": 45.46,
        "n_payments": 1.0,
        "max_installments": 1.0,
        "order_purchase_timestamp": "2018-05-15 10:00:00",
        "order_estimated_delivery_date": "2018-05-25 00:00:00",
    }
