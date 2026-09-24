from typing import List

from pydantic import BaseModel, ConfigDict, Field


class OrderInput(BaseModel):
    customer_state: str
    customer_zip_code_prefix: int = Field(ge=0, le=99999)
    n_items: float = Field(ge=0)
    total_price: float = Field(ge=0)
    total_freight: float = Field(ge=0)
    total_payment_value: float = Field(ge=0)
    n_payments: float = Field(ge=0)
    max_installments: float = Field(ge=0)
    order_purchase_timestamp: str
    order_estimated_delivery_date: str


class PredictionOutput(BaseModel):
    prediction: int
    is_late: bool
    probability: float
    model_version: str

    model_config = ConfigDict(protected_namespaces=())


class BatchOrderInput(BaseModel):
    orders: List[OrderInput] = Field(min_length=1)


class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
