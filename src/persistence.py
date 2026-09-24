"""Durable prediction logging for deployments with PostgreSQL."""

import json
from datetime import datetime, timezone

from src.config import config
from src.logger import logger


CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    input_payload JSONB NOT NULL,
    prediction INTEGER NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    model_version TEXT NOT NULL,
    latency_seconds DOUBLE PRECISION NOT NULL
)
"""


def persist_prediction(payload: dict, result: dict, latency_seconds: float) -> bool:
    database_url = config["storage"].get("database_url", "")
    if not database_url:
        return False
    try:
        import psycopg

        with psycopg.connect(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(CREATE_TABLE)
                cursor.execute(
                    """
                    INSERT INTO prediction_logs
                    (created_at, input_payload, prediction, probability, model_version, latency_seconds)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        datetime.now(timezone.utc),
                        json.dumps(payload),
                        result["prediction"],
                        result["probability"],
                        result["model_version"],
                        latency_seconds,
                    ),
                )
        return True
    except Exception as error:
        logger.warning("prediction persistence failed: %s", error)
        return False