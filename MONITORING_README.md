# Monitoring & Logging (Task 8)

## What's already built in

Every request through `api/main.py` logs a structured line (request ID, method, path, status, duration) to stdout — visible via `docker logs -f heart-disease-api` or `kubectl logs`. You already captured this in your Docker screenshots.

On top of that, the API exposes a **Prometheus-format `/metrics` endpoint** with:
- Default HTTP metrics (`http_requests_total`, request latency histograms) — broken down by endpoint, method, status code
- A custom `heart_disease_predictions_total` counter, labeled by predicted class — lets you chart the model's positive-rate over time, which is the simplest possible data-drift signal in production

## Running the full Prometheus + Grafana stack locally

Everything's already wired into `docker-compose.yml`:

```bash
docker-compose up --build -d
```

This starts three containers on a shared network:
- `heart-disease-api` — the model API (port 8000)
- `prometheus` — scrapes `/metrics` from the API every 5s (port 9090)
- `grafana` — pre-provisioned dashboard reading from Prometheus (port 3000)

## Viewing it

1. **Prometheus** — http://localhost:9090 → Status → Targets, confirm `heart-disease-api` shows as `UP`. You can also run ad-hoc queries here, e.g. `rate(http_requests_total[1m])`.
2. **Grafana** — http://localhost:3000, log in as `admin` / `admin` (or just browse anonymously — it's enabled for convenience). The **"Heart Disease API — Monitoring"** dashboard is pre-loaded automatically (no manual setup needed) with 4 panels:
   - Request rate (req/s) by endpoint
   - p95 request latency
   - Predictions by class (0 vs 1) over time
   - 5xx error rate

Send a few requests to `/predict` first (or just refresh Swagger UI a few times) so the graphs have something to show.

## Screenshot checklist for your report

- Prometheus targets page showing the API as `UP`
- Grafana dashboard with at least the request-rate and predictions-by-class panels populated
- A terminal showing `docker-compose ps` with all three containers healthy
