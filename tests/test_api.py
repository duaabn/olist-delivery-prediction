from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_and_model_info():
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.headers["x-request-id"]
    info = client.get("/info").json()
    assert info["model_version"] == "v2.0.0"
    assert info["feature_count"] == 42


def test_readiness_and_prometheus_metrics():
    assert client.get("/ready").json()["status"] == "ready"
    metrics_response = client.get("/metrics/prometheus")
    assert metrics_response.status_code == 200
    assert "olist_http_responses_total" in metrics_response.text


def test_predict_and_batch(sample_order):
    single = client.post("/predict", json=sample_order)
    batch = client.post("/predict_batch", json={"orders": [sample_order]})
    assert single.status_code == 200
    assert batch.status_code == 200
    assert batch.json()["predictions"][0] == single.json()


def test_bad_payload_is_rejected(sample_order):
    response = client.post("/predict", json={**sample_order, "n_items": -1})
    assert response.status_code == 422


def test_schema_error_has_standard_shape(sample_order):
    response = client.post("/predict", json={**sample_order, "n_items": "invalid"})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert body["request_id"] == response.headers["x-request-id"]


def test_metrics_are_exposed():
    metrics = client.get("/metrics").json()
    assert metrics["requests"] >= 1
    assert "error_rate" in metrics
