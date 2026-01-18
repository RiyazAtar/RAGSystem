# Marketing AI - Deployment Guide

This guide explains how to deploy the Marketing AI system using automated scripts.

## 📋 Prerequisites

- Docker Desktop installed and running
- Python 3.11+ installed (for training)
- 4GB+ RAM available
- Data file: `data/amazon_sales.csv`

## 🚀 Quick Start - Complete Deployment

Run everything with one command:

```bash
./deploy.sh
```

This automated script will:
1. ✅ Check prerequisites (Docker, data file)
2. 🧠 Train the ML model
3. 🧪 Run unit tests
4. 🚀 Deploy the main application server
5. 📊 Deploy monitoring stack (Prometheus + Grafana)

## 📜 Available Scripts

### 1. `train.sh` - Train Model Only

Trains the discount prediction model and RAG system.

```bash
./train.sh
```

**What it does:**
- Creates `models/` directory
- Installs Python dependencies
- Runs `train.py`
- Saves model artifacts:
  - `models/discount_model.joblib`
  - `models/processor.joblib`
  - `models/rag_system/`

**Duration:** ~2-5 minutes depending on data size

---

### 2. `test.sh` - Run Unit Tests

Runs all unit tests with coverage reporting.

```bash
./test.sh
```

**What it does:**
- Installs pytest and dependencies
- Creates basic test structure if missing
- Runs all tests in `tests/` directory
- Generates coverage report
- Saves results to `test_results.txt`

**Output:**
- ✓ Pass/fail status for each test
- Code coverage percentage
- Missing coverage lines

---

### 3. `deploy.sh` - Complete Deployment

Master script that automates the entire deployment pipeline.

```bash
./deploy.sh
```

**What it does:**

**Step 1: Prerequisites Check**
- Verifies Docker is running
- Verifies data file exists

**Step 2: Train Model**
- Calls `train.sh`

**Step 3: Run Tests**
- Calls `test.sh`
- Exits if tests fail

**Step 4: Deploy Main Server**
- Stops existing containers
- Builds Docker image from `Dockerfile`
- Starts container via `docker-compose.yml`
- Waits for health check
- Exposes ports:
  - 8000: FastAPI application
  - 8001: Prometheus metrics

**Step 5: Deploy Monitoring**
- Starts Prometheus container (port 9090)
- Starts Grafana container (port 3000)
- Waits for health checks
- Configures metric scraping

**Duration:** ~10-15 minutes total

---

### 4. `stop.sh` - Stop All Services

Stops all running Docker containers.

```bash
./stop.sh
```

**What it does:**
- Stops main application container
- Stops Prometheus container
- Stops Grafana container
- Removes containers (preserves volumes)

---

## 🌐 Service URLs (After Deployment)

### Main Application
- **API**: http://localhost:8000
- **UI**: http://localhost:8000/ui/index.html
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Internal Metrics**: http://localhost:8000/metrics

### Monitoring Stack
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `admin`
- **Prometheus Metrics**: http://localhost:8001/metrics

---

## 🔧 Common Operations

### View Logs

**Application logs:**
```bash
docker compose logs -f
```

**Monitoring logs:**
```bash
docker compose -f docker-compose.monitoring.yml logs -f
```

**Specific container:**
```bash
docker logs -f marketing-ai
docker logs -f marketing-ai-prometheus
docker logs -f marketing-ai-grafana
```

### Restart Services

**Restart application only:**
```bash
docker compose restart
```

**Restart everything:**
```bash
./stop.sh
./deploy.sh
```

### Check Container Status

```bash
docker ps
```

### Access Container Shell

```bash
docker exec -it marketing-ai bash
```

### Remove Everything (Clean Slate)

```bash
docker compose down -v
docker compose -f docker-compose.monitoring.yml down -v
rm -rf models/
```

---

## 🧪 Testing Endpoints

After deployment, test the APIs:

### Health Check
```bash
curl http://localhost:8000/health
```

### Predict Discount
```bash
curl -X POST http://localhost:8000/predict_discount \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Test Headphones",
    "category": "Electronics|Headphones",
    "actual_price": 2990,
    "rating": 4.1,
    "rating_count": 1000
  }'
```

### RAG Query
```bash
curl -X POST http://localhost:8000/answer_question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Show me headphones under 3000 rupees",
    "session_id": "test-session"
  }'
```

### Check Prometheus Metrics
```bash
curl http://localhost:8001/metrics | grep discount_predictions_total
```

---

## 🐛 Troubleshooting

### Docker not running
**Error:** `Cannot connect to the Docker daemon`

**Solution:**
```bash
# Start Docker Desktop manually, then run:
./deploy.sh
```

### Port already in use
**Error:** `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Solution:**
```bash
# Find and kill process using the port
lsof -ti:8000 | xargs kill -9

# Or use a different port in docker-compose.yml
```

### Container unhealthy
**Error:** `Service did not become healthy within timeout`

**Solution:**
```bash
# Check logs for errors
docker-compose logs

# Common fixes:
# 1. Ensure data file exists: data/amazon_sales.csv
# 2. Ensure enough disk space
# 3. Rebuild image: docker-compose build --no-cache
```

### Model not found
**Error:** `Model file not found: models/discount_model.joblib`

**Solution:**
```bash
# Train model first
./train.sh
```

### Tests failing
**Error:** Tests fail during deployment

**Solution:**
```bash
# Run tests independently to see details
./test.sh

# Check test_results.txt for details
cat test_results.txt
```

---

## 📊 Monitoring Setup

After deployment, configure Grafana:

1. Open http://localhost:3000
2. Login: `admin` / `admin`
3. Add Prometheus data source:
   - URL: `http://prometheus:9090`
   - Access: Server (default)
4. Import dashboard (grafana_dashboard.json)
5. View real-time metrics

---

## 🔄 CI/CD Integration

To integrate with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Deploy Marketing AI
  run: |
    chmod +x deploy.sh
    ./deploy.sh
```

---

## 📁 Project Structure

```
marketing-ai/
├── deploy.sh              # Master deployment script
├── train.sh               # Model training script
├── test.sh                # Unit testing script
├── stop.sh                # Stop all services script
├── docker-compose.yml     # Main app configuration
├── docker-compose.monitoring.yml  # Monitoring configuration
├── Dockerfile             # Container image definition
├── train.py               # Training script
├── src/                   # Source code
│   ├── api.py            # FastAPI application
│   ├── discount_model.py # ML model
│   ├── rag_system.py     # RAG Q&A system
│   └── monitoring.py     # Prometheus metrics
├── data/                  # Data files
│   └── amazon_sales.csv
├── models/                # Trained models (generated)
├── tests/                 # Unit tests
└── ui/                    # Frontend files
```

---

## 🎯 Production Deployment Notes

For production deployment, consider:

1. **Environment Variables**: Use `.env` file for secrets
2. **Resource Limits**: Adjust in `docker-compose.yml`
3. **Persistent Storage**: Use named volumes for data
4. **Load Balancing**: Deploy multiple replicas
5. **HTTPS**: Add reverse proxy (nginx/traefik)
6. **Logging**: Integrate with centralized logging
7. **Alerting**: Configure Grafana alerts
8. **Backups**: Automate model and data backups

---
