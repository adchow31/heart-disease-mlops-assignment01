# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# Avoid .pyc files and enable unbuffered stdout (so logs stream to `docker logs`)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install only what's needed for the API at runtime (skip matplotlib/seaborn/
# jupyter/mlflow-training deps to keep the image lean)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code and the trained model artifacts
COPY api/ ./api/
COPY models/ ./models/

# Non-root user for defense-in-depth
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
