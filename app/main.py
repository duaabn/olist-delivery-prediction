import time
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.schemas import (
    BatchOrderInput,
    BatchPredictionOutput,
    OrderInput,
    PredictionOutput,
)
from src.config import config
from src.data_validation import validate_with_great_expectations
from src.logger import logger
from src.metrics import metrics
from src.persistence import persist_prediction
from src.predict import make_batch_prediction, make_prediction, model_info as loaded_model_info
from src.preprocess import apply_preprocessing
from src.validation import validate_orders


app = FastAPI(
    title="Olist Delivery Prediction API",
    version=config["model"]["version"],
    description="Production inference service for late-delivery prediction.",
)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            metrics.record_http(500)
            raise
        latency = time.perf_counter() - started
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{latency:.6f}"
        metrics.record_http(response.status_code)
        logger.info(
            "http_request request_id=%s method=%s path=%s status=%d latency_seconds=%.6f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency,
        )
        return response


app.add_middleware(RequestContextMiddleware)


@app.exception_handler(RequestValidationError)
async def request_validation_error(request: Request, error: RequestValidationError):
    request_id = getattr(request.state, "request_id", request.headers.get("X-Request-ID", "unknown"))
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Request payload failed validation",
            "details": [
                {
                    "location": list(item.get("loc", [])),
                    "message": str(item.get("msg", "Invalid value")),
                    "type": str(item.get("type", "validation_error")),
                }
                for item in error.errors()
            ],
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "model_version": config["model"]["version"]}


@app.get("/ready")
def readiness_check():
    required_paths = [
        config["paths"]["model_path"],
        config["paths"]["imputer_path"],
        config["paths"]["encoder_path"],
        config["paths"]["scaler_path"],
        config["paths"]["feature_list_path"],
    ]
    missing = [path for path in required_paths if not Path(path).is_file()]
    if missing:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "missing": missing})
    return {"status": "ready", "model_version": config["model"]["version"]}


@app.get("/info")
def model_info():
    return loaded_model_info()


@app.get("/metrics")
def service_metrics():
    return metrics.snapshot()


@app.get("/metrics/prometheus", response_class=PlainTextResponse)
def prometheus_metrics():
    return metrics.prometheus()


@app.post("/predict", response_model=PredictionOutput)
def predict_single(order: OrderInput):
    started = time.perf_counter()
    payload = order.model_dump()
    try:
        df = pd.DataFrame([payload])
        validate_orders(df)
        validate_with_great_expectations(df)
        result = make_prediction(apply_preprocessing(df))
        latency = time.perf_counter() - started
        metrics.record(latency, result["prediction"])
        persist_prediction(payload, result, latency)
        logger.info("prediction input=%s output=%s latency_seconds=%.6f", payload, result, latency)
        return result
    except (ValueError, KeyError, TypeError) as error:
        latency = time.perf_counter() - started
        metrics.record(latency, error=True)
        logger.warning("prediction rejected input=%s error=%s", payload, error)
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/predict_batch", response_model=BatchPredictionOutput)
def predict_batch(batch: BatchOrderInput):
    started = time.perf_counter()
    payload = [order.model_dump() for order in batch.orders]
    try:
        df = pd.DataFrame(payload)
        validate_orders(df)
        validate_with_great_expectations(df)
        results = make_batch_prediction(apply_preprocessing(df))
        latency = time.perf_counter() - started
        for index, result in enumerate(results):
            metrics.record(latency, result["prediction"])
            persist_prediction(payload[index], result, latency)
        logger.info("batch_prediction size=%d outputs=%s latency_seconds=%.6f", len(payload), results, latency)
        return {"predictions": results}
    except (ValueError, KeyError, TypeError) as error:
        latency = time.perf_counter() - started
        metrics.record(latency, error=True)
        logger.warning("batch_prediction rejected size=%d error=%s", len(payload), error)
        raise HTTPException(status_code=422, detail=str(error)) from error
