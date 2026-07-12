import json
import os

import joblib
import pandas as pd
import pytest

MODEL_PATH = os.path.join("models", "heart_disease_model.joblib")
METADATA_PATH = os.path.join("models", "model_metadata.json")


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model artifact not found at {MODEL_PATH}; run the notebook first.")
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def metadata():
    if not os.path.exists(METADATA_PATH):
        pytest.skip(f"Metadata not found at {METADATA_PATH}; run the notebook first.")
    with open(METADATA_PATH) as f:
        return json.load(f)


@pytest.fixture
def sample_row(metadata):
    example = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }
    return pd.DataFrame([example])[metadata["feature_columns"]]


def test_model_loads(model):
    assert model is not None
    assert hasattr(model, "predict") and hasattr(model, "predict_proba")


def test_metadata_matches_expected_schema(metadata):
    expected = {"age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal"}
    assert set(metadata["feature_columns"]) == expected
    assert "roc_auc" in metadata["metrics"]


def test_predict_returns_valid_class(model, sample_row):
    pred = model.predict(sample_row)
    assert pred[0] in (0, 1)


def test_predict_proba_sums_to_one(model, sample_row):
    proba = model.predict_proba(sample_row)[0]
    assert len(proba) == 2
    assert abs(sum(proba) - 1.0) < 1e-6
    assert all(0.0 <= p <= 1.0 for p in proba)


def test_predictions_are_deterministic(model, sample_row):
    p1 = model.predict_proba(sample_row)[0, 1]
    p2 = model.predict_proba(sample_row)[0, 1]
    assert p1 == p2


def test_model_meets_minimum_quality_bar(metadata):
    # Regression guard: fail CI if a future retrain quietly tanks quality
    assert metadata["metrics"]["roc_auc"] >= 0.75
    assert metadata["metrics"]["accuracy"] >= 0.70
