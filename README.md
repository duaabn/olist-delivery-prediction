# Olist Delivery Prediction

Production inference service for the Olist late-delivery classifier. Training remains in the notebooks; the service only loads the fitted model, imputer, encoder, scaler, and feature list from `models/`.

## Run from zero

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m pytest -q
uvicorn app.main:app --reload
```

The API is available at `http://localhost:8000`; interactive documentation is at `/docs`.

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/info
```

Send the ten prediction-time fields, including `order_estimated_delivery_date`, to `POST /predict`. `POST /predict_batch` accepts `{ "orders": [...] }`. Both responses contain `prediction`, `is_late`, `probability`, and `model_version`.

## Structure

- `app/`: FastAPI routes and Pydantic request/response schemas.
- `src/`: configuration, validation, fitted preprocessing, prediction, logging, and metrics.
- `config/`: YAML configuration; paths are resolved relative to the project root.
- `models/`: versioned production artifacts promoted from the notebook output.
- `tests/`: preprocessing, model, validation, and API integration tests.
- `notebooks/`: training and exploration only; never imported by the service.
- `data/`: DVC-managed data locations.

## Reproducible artifacts

Notebook 5 and 6 produce the artifacts. Copy/promote the selected artifacts into `models/`, update `model.version` in `config/config.yaml`, and run `pytest -q`. `dvc.yaml` gives DVC a reproducible artifact stage; configure a remote with `dvc remote add` before sharing artifacts. Do not commit secrets or local `.dvc/cache` files.

The production `models/` directory is tracked by DVC in `models.dvc`. The configured default remote is local storage at `.dvc/local-remote`, which keeps this project reproducible without OAuth or external credentials. Upload and verify it with:

```powershell
dvc repro artifacts
dvc push
dvc status
```

The local remote is intentionally ignored by Git. Commit `models.dvc`, `dvc.yaml`, and `dvc.lock`; do not commit `.dvc/local-remote` or `.dvc/cache`.

## Containers

```powershell
docker compose up --build
```

Compose starts the API, PostgreSQL for prediction-log storage integration, and MinIO as the artifact-store endpoint. Credentials and connection values come from environment variables; use a private `.env` file locally.

## Tracking and monitoring

MLflow is pinned in `requirements.txt` for registering the selected notebook model and retaining parameters, metrics, and artifacts. A deployment should set the model version in `config/config.yaml` only after the MLflow run has been approved and its artifacts have been promoted to the configured artifact store.

To record a promotion in MLflow, provide a reachable tracking server and run the promotion command with `MLFLOW_ENABLED=true` and `MLFLOW_TRACKING_URI` set. Without those variables, promotion remains local and deterministic.

The API uses fast local schema/range validation by default. Set `GE_ENABLED=true` when the Great Expectations runtime is required in a deployment; Compose enables it by default.

The `/metrics` endpoint exposes request count, error rate, average latency, and prediction counts. Structured logs in `logs/prediction.log` include input, output, latency, and model version. Alert on sustained error rate above 5%, p95 latency above the agreed SLO, or a material shift in the late/on-time distribution. Actual delivery outcomes should be joined later for drift and delayed-label evaluation.

Compose also starts Prometheus at `http://localhost:9090` and Grafana at `http://localhost:3000`. Grafana is provisioned with the Prometheus datasource and an Olist API dashboard; change the default admin password through `GRAFANA_ADMIN_PASSWORD`.

## CI/CD

Every push and pull request runs Ruff, formatting checks, pytest, and a Docker build. Install pre-commit once with `pre-commit install` so the same style checks run before commits.
