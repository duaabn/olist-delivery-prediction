from pathlib import Path
import os

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path=None):
    """Reads the YAML configuration file."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "config.yaml"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["paths"] = {
        "model_path": str(PROJECT_ROOT / config["model"]["path"]),
        "imputer_path": str(PROJECT_ROOT / config["transformers"]["imputer_path"]),
        "encoder_path": str(PROJECT_ROOT / config["transformers"]["encoder_path"]),
        "scaler_path": str(PROJECT_ROOT / config["transformers"]["scaler_path"]),
        "feature_list_path": str(PROJECT_ROOT / config["transformers"]["feature_list_path"]),
        "log_file": str(PROJECT_ROOT / config["logging"]["log_file"]),
    }
    config["features"]["categorical_cols"] = [config["features"]["categorical_col"]]
    config["model"]["version"] = os.getenv("MODEL_VERSION", config["model"]["version"])
    config["storage"]["database_url"] = os.getenv(
        "DATABASE_URL", config["storage"].get("database_url", "")
    )
    config["tracking"]["enabled"] = os.getenv(
        "MLFLOW_ENABLED", str(config["tracking"].get("enabled", False))
    ).lower() in {"1", "true", "yes"}
    config["tracking"]["uri"] = os.getenv("MLFLOW_TRACKING_URI", config["tracking"].get("uri", ""))
    config["validation"]["great_expectations_enabled"] = os.getenv(
        "GE_ENABLED", str(config["validation"].get("great_expectations_enabled", False))
    ).lower() in {"1", "true", "yes"}
    return config

config = load_config() 