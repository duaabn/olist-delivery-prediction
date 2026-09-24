import logging

import pandas as pd

from src.predict import make_prediction
from src.preprocess import apply_preprocessing


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


sample_order = pd.DataFrame([{
    "customer_state": "SP",
    "customer_zip_code_prefix": 14600,
    "n_items": 1.0,
    "total_price": 29.90,
    "total_freight": 15.56,
    "total_payment_value": 45.46,
    "n_payments": 1.0,
    "max_installments": 1.0,
    "order_purchase_timestamp": "2018-05-15 10:00:00",
    "order_estimated_delivery_date": "2018-05-25 00:00:00",
}])


if __name__ == "__main__":
    processed_data = apply_preprocessing(sample_order)
    result = make_prediction(processed_data)
    logger.info("processed_shape=%s prediction=%s", processed_data.shape, result)
