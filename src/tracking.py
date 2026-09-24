"""Optional MLflow tracking for approved model promotions."""

from pathlib import Path

from src.config import config
from src.logger import logger


def track_promotion(metrics: dict, model, artifact_paths: list[Path]) -> str | None:
    if not config["tracking"].get("enabled"):
        return None
    import mlflow
    import mlflow.sklearn

    uri = config["tracking"].get("uri")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config["tracking"]["experiment_name"])
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", metrics["model"])
        mlflow.log_param("decision_threshold", metrics["threshold"])
        mlflow.log_metrics({key: value for key, value in metrics.items() if key.startswith(("test_", "validation_"))})
        for artifact_path in artifact_paths:
            mlflow.log_artifact(str(artifact_path))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=config["tracking"]["registered_model_name"],
        )
        logger.info("MLflow model promotion run_id=%s", run.info.run_id)
        return run.info.run_id