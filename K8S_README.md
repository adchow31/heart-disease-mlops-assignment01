# Kubernetes Deployment (Task 7)

You said you're targeting **local only (Docker Desktop + Minikube)** — here's how to do it either way. Pick whichever cluster you have enabled.

## Option A — Docker Desktop Kubernetes

1. Docker Desktop → Settings → Kubernetes → check "Enable Kubernetes" → Apply & Restart. Wait for the green "Kubernetes running" indicator.
2. Build the image (same command as before — Docker Desktop's Kubernetes shares the same image store as `docker build`):
   ```bash
   docker build -t heart-disease-api:latest .
   ```
3. Apply the manifests:
   ```bash
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   ```
4. Check status:
   ```bash
   kubectl get pods -n mlops-assignment
   kubectl get svc -n mlops-assignment
   ```
5. On Docker Desktop, `LoadBalancer` services resolve to `localhost` directly:
   ```bash
   curl http://localhost/health
   ```
   (Service listens on port 80 → forwards to container port 8000.)

## Option B — Minikube

1. Start Minikube if it isn't already running:
   ```bash
   minikube start
   ```
2. **Important:** Minikube runs its own separate Docker daemon, so a plain `docker build` on your host won't be visible to it. Either:
   - Point your shell at Minikube's Docker daemon before building:
     ```bash
     eval $(minikube docker-env)      # Linux/Mac
     minikube docker-env | Invoke-Expression   # Windows PowerShell
     docker build -t heart-disease-api:latest .
     ```
   - **or** build normally and load the image in:
     ```bash
     docker build -t heart-disease-api:latest .
     minikube image load heart-disease-api:latest
     ```
3. Apply the manifests (same as Option A):
   ```bash
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   ```
4. Minikube doesn't provide a real cloud load balancer, so use its tunnel helper to expose the service:
   ```bash
   minikube service heart-disease-api-service -n mlops-assignment
   ```
   This opens the service in your browser (or prints the URL) automatically.

## Verifying the deployment (for your report screenshots)

```bash
kubectl get all -n mlops-assignment
kubectl describe deployment heart-disease-api -n mlops-assignment
kubectl logs -n mlops-assignment -l app=heart-disease-api --tail=50
```

You should see **2 running pods** (set via `replicas: 2` in `deployment.yaml`) — this demonstrates the deployment can survive a single pod crashing, which is worth mentioning in your report's architecture section.

## What the manifests do

- **`deployment.yaml`** — creates the `mlops-assignment` namespace, then a Deployment with 2 replicas, resource requests/limits (100m–500m CPU, 256Mi–512Mi memory), and readiness/liveness probes hitting `/health` — so Kubernetes won't route traffic to a pod until the model has actually finished loading, and will restart a pod automatically if it stops responding.
- **`service.yaml`** — a `LoadBalancer` Service mapping external port 80 → container port 8000, so you don't need to remember the `:8000` in URLs.

## Cleaning up

```bash
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deployment.yaml
```
