# Qafza Training Alignment

This project follows the engineering rules demonstrated in the Qafza Free Training repository. It uses the ideas, not copied course code.

| Qafza session | Project implementation | Evidence |
|---|---|---|
| 01 Leakage-proof pipeline | Prediction-time fields are explicit; delivery outcome and review fields are excluded. | `src/preprocess.py`, `src/promote_model.py` |
| 03 Production API | FastAPI, Pydantic contracts, structured errors, request IDs, readiness, and batch inference. | `app/`, `tests/test_api.py` |
| 04 Docker | Small runtime image, Compose services, healthchecks, environment variables. | `Dockerfile`, `docker-compose.yml` |
| 05 ETL principles | Data access, validation, feature construction, and prediction are separate modules. | `src/validation.py`, `src/preprocess.py` |
| 06 MLflow | Promotion can log parameters, metrics, model, and artifacts when enabled. | `src/tracking.py`, `src/promote_model.py` |
| 07 DVC | Production artifact set is declared and missing artifacts fail the DVC stage. | `dvc.yaml` |
| 10 Monitoring | Prometheus endpoint, Prometheus scrape config, Grafana datasource and dashboard, logs, and drift-oriented prediction counts. | `src/metrics.py`, `monitoring/` |
| 11 Automation | CI runs lint, format, tests, and image build; pre-commit runs the same code-quality checks. | `.github/workflows/ci.yml`, `.pre-commit-config.yaml` |
| 13 End-to-end | A single service connects versioned artifacts, inference, API, Docker, CI, tracking, persistence, and monitoring. | `README.md` |

## Deliberately excluded sessions

Sessions 02 (deep learning), 08 (distributed Ray training), 09 (Feast feature store), and 12 (Terraform infrastructure) are not forced into this inference-only project. They would add infrastructure without solving a requirement here. They become appropriate when the project introduces neural models, distributed training, online feature serving, or a cloud deployment target.

## Verification rule

No claim is considered complete until a command proves it:

```powershell
pytest -q
docker compose config
docker build -t olist-delivery-prediction:local .
```

The test suite must fail the CI job when a contract, artifact, or API behavior is broken.