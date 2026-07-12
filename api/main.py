import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import HealthResponse, PatientFeatures, PredictionResponse

# Custom business-level metric (on top of the instrumentator's default
# request-count / latency / status-code metrics) tracking prediction
# outcomes, so Grafana can chart e.g. positive-rate drift over time.
PREDICTION_COUNTER = Counter(
    "heart_disease_predictions_total",
    "Total predictions made, labeled by predicted class",
    ["prediction_label"],
)

# ---------------------------------------------------------------------------
# Logging: structured-ish request/response logging to stdout (picked up by
# `docker logs` / any log aggregator) and to a rotating file for local runs.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("heart-disease-api")

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "heart_disease_model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

_state = {"model": None, "metadata": None}


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
    return model, metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model from %s", MODEL_PATH)
    _state["model"], _state["metadata"] = load_model()
    logger.info("Model loaded: %s", _state["metadata"].get("model_name", "unknown"))
    yield
    _state["model"] = None


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts heart disease risk from patient health data (UCI Heart Disease dataset).",
    version="1.0.0",
    lifespan=lifespan,
)

# Exposes GET /metrics in Prometheus text format: request counts, latency
# histograms, and in-progress requests, broken down by handler/method/status.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id, request.method, request.url.path, response.status_code, duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/", tags=["meta"])
def root():
    return {"service": "heart-disease-risk-api", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    model_loaded = _state["model"] is not None
    return HealthResponse(
        status="ok" if model_loaded else "unavailable",
        model_loaded=model_loaded,
        model_name=(_state["metadata"] or {}).get("model_name"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(features: PatientFeatures):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metadata = _state["metadata"] or {}
    feature_columns = metadata.get("feature_columns", list(features.model_dump().keys()))

    row = pd.DataFrame([features.model_dump()])[feature_columns]

    try:
        proba = _state["model"].predict_proba(row)[0]
        pred = int(proba[1] >= 0.5)
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}")

    confidence = float(proba[pred])
    label_map = metadata.get("target_mapping", {"0": "no heart disease", "1": "heart disease present"})
    PREDICTION_COUNTER.labels(prediction_label=str(pred)).inc()

    return PredictionResponse(
        prediction=pred,
        label=label_map.get(str(pred), str(pred)),
        confidence=confidence,
        probability_disease=float(proba[1]),
        model_name=metadata.get("model_name", "unknown"),
        model_version=metadata.get("trained_at_utc", "unknown"),
    )
