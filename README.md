# Heart Disease Risk Prediction — MLOps Pipeline

End-to-end MLOps pipeline for heart disease risk prediction, built for **Machine Learning Operations (MLOps) AIMLCZG523, Assignment 01** (BITS Pilani WILP).

Predicts heart disease risk from patient health data (UCI Heart Disease — Cleveland dataset) and serves it as a monitored, containerized, Kubernetes-deployable API.

**Best model:** Random Forest — 88.5% accuracy, 0.947 ROC-AUC (see the notebook and report for full metrics).

## Project Structure

```
.
├── MLOps_Assignment_01_Aditya_Chowdhury_2024AD05010.ipynb   # EDA, modeling, MLflow tracking, packaging
├── Heart_Disease_MLOps_Assignment01_Report.docx / .pdf       # Final written report
├── data/                    # Raw + cleaned dataset (CSV)
├── plots/                   # EDA and model evaluation plots
├── models/                  # Saved pipeline (.joblib) + metadata (.json)
├── mlflow.db                # MLflow experiment tracking (SQLite backend)
├── api/                     # FastAPI serving app (main.py, schemas.py)
├── tests/                   # Pytest suite (data processing, model, API)
├── scripts/train.py         # Standalone/CI-friendly training script
├── Dockerfile, docker-compose.yml, requirements*.txt
├── k8s/                     # Kubernetes Deployment + Service manifests
├── monitoring/              # Prometheus config + Grafana provisioning
├── .github/workflows/       # CI/CD pipeline (GitHub Actions)
└── DOCKER_README.md, K8S_README.md, MONITORING_README.md      # Setup guides per component
```

## Quick Start

### 1. Train the model
Run the notebook top to bottom, or headlessly:
```bash
pip install -r requirements.txt
python scripts/train.py
```
Produces `models/heart_disease_model.joblib` and `models/model_metadata.json`.

### 2. Run the API + monitoring stack
```bash
docker-compose up --build -d
```
- API: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

Full details: [DOCKER_README.md](DOCKER_README.md), [MONITORING_README.md](MONITORING_README.md)

### 3. Deploy to Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```
Full details: [K8S_README.md](K8S_README.md)

### 4. Run tests
```bash
pytest tests/ -v
```

## CI/CD

Every push to `main` triggers `.github/workflows/ci-cd.yml`: **lint → train → test → docker-build**, with artifacts (trained model, coverage report, smoke-test logs) uploaded per run. See the Actions tab for run history.

## Tech Stack

Python · scikit-learn · XGBoost · MLflow · FastAPI · Docker · Kubernetes · GitHub Actions · Prometheus · Grafana

## Author

Aditya Chowdhury — BITS ID 2024AD05010
