import os

import pytest
from fastapi.testclient import TestClient

MODEL_PATH = os.path.join("models", "heart_disease_model.joblib")

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Model artifact not found; run the notebook first.",
)

VALID_PAYLOAD = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
}


@pytest.fixture
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_payload(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert body["label"] in ("no heart disease", "heart disease present")


def test_predict_missing_field_returns_422(client):
    bad_payload = dict(VALID_PAYLOAD)
    del bad_payload["age"]
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_predict_out_of_range_value_returns_422(client):
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["age"] = 500  # outside allowed ge/le bounds
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_predict_wrong_type_returns_422(client):
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["sex"] = "not-a-number"
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_response_has_request_id_header(client):
    r = client.get("/health")
    assert "X-Request-ID" in r.headers


def test_metrics_endpoint_exposes_prometheus_format(client):
    client.post("/predict", json=VALID_PAYLOAD)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "heart_disease_predictions_total" in r.text
    assert "http_requests_total" in r.text