# Running the Heart Disease API in Docker

## Prerequisites
- Docker Desktop running
- The trained model must already exist at `models/heart_disease_model.joblib`
  and `models/model_metadata.json` (produced by running the notebook first).

## Build & run

```bash
# from the project root
docker build -t heart-disease-api:latest .
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest
```

or with docker-compose:

```bash
docker-compose up --build -d
```

## Verify it's working

```bash
# health check
curl http://localhost:8000/health
# -> {"status":"ok","model_loaded":true,"model_name":"random_forest"}

# prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
      }'
```

Or open **http://localhost:8000/docs** for the interactive Swagger UI (FastAPI auto-generates this) — good for a "deployment screenshot" in the report.

## Logs

```bash
docker logs -f heart-disease-api
```

Every request is logged with a request ID, method, path, status code, and duration — this satisfies the assignment's "API request logging" requirement and is what Prometheus/Grafana or a log dashboard would tail in the Monitoring step.

## Stop

```bash
docker-compose down
# or
docker stop heart-disease-api && docker rm heart-disease-api
```
