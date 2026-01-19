# Marketing AI - Deployment Scripts

This directory contains automated deployment scripts for the Marketing AI system.

## 🚀 Quick Start

From the **project root** directory, run:

```bash
# Complete automated deployment (recommended)
scripts/deploy.sh
```

## 📜 Available Scripts

### 1. `deploy.sh` - Complete Deployment Pipeline ⭐

**Master script** that automates everything:

```bash
scripts/deploy.sh
```

**What it does:**
1. ✅ Checks Docker is running
2. ✅ Checks data file exists
3. 🧠 Trains ML model (`scripts/train.sh`)
4. 🧪 Runs unit tests (`scripts/test.sh`)
5. 🚀 Deploys main application (port 8000, 8001)
6. 📊 Deploys monitoring (Prometheus 9090, Grafana 3000)
7. ⏱️ Waits for health checks
8. 📋 Shows all service URLs

**Output includes:* 
- Real-time progress updates
- Health check verification
- Service URLs and commands

---

### 2. `train.sh` - Train Model Only

Trains the discount prediction model and RAG system:

```bash
scripts/train.sh
```

**What it does:**
- Checks `data/amazon_sales.csv` exists
- Creates `models/` directory
- Installs Python dependencies
- Runs `train.py`
- Saves model artifacts

**Output:**
- `models/discount_model.joblib`
- `models/processor.joblib`
- `models/rag_system/`

---

### 3. `test.sh` - Run Unit Tests

Runs all unit tests with coverage:

```bash
scripts/test.sh
```

**What it does:**
- Installs pytest dependencies
- Creates test structure if missing
- Runs all tests in `tests/`
- Generates coverage report
- Saves results to `test_results.txt`

**Creates (if missing):**
- `tests/` directory
- `tests/test_api.py`
- `tests/__init__.py`

---

### 4. `stop.sh` - Stop All Services

Stops all Docker containers:

```bash
scripts/stop.sh
```

**What it does:**
- Stops main application container
- Stops Prometheus container
- Stops Grafana container
- Removes containers (preserves data volumes)

---

### 5. `view_conversations.sh` - View Conversation History

Interactive tool to query the conversation database:

```bash
scripts/view_conversations.sh
```

**What it does:**
- Lists all conversation sessions
- View detailed messages for a session
- Show database statistics
- Search messages by content

**Menu Options:**
1. List all sessions (shows recent 20)
2. View session details (enter session_id)
3. Show statistics (total messages, avg per session)
4. Search messages (search by keyword)
5. Exit

**Example Output:**
```
Recent Sessions:
session_id          msgs  created              last_active
user_k7m2x9w1q     12    2025-01-17 10:30:00  2025-01-17 10:45:00
user_p3j8m2r5t     8     2025-01-17 09:15:00  2025-01-17 09:30:00
```

---

## 🎯 Usage Examples

### Full Deployment
```bash
# From project root
cd /path/to/marketing-ai
scripts/deploy.sh
```

### Individual Steps
```bash
# Train model only
scripts/train.sh

# Run tests only
scripts/test.sh

# Stop everything
scripts/stop.sh
```

### Redeploy After Code Changes
```bash
# Stop, rebuild, and restart
scripts/stop.sh
scripts/deploy.sh
```

---

## 📂 How Scripts Work

All scripts use **relative path resolution** to work from anywhere:

```bash
# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get project root (parent of scripts/)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"
```

This means:
- ✅ Scripts work when called from project root: `scripts/deploy.sh`
- ✅ Scripts work when called from scripts dir: `./deploy.sh`
- ✅ Scripts work from any directory: `/full/path/to/scripts/deploy.sh`

---

## 🔧 Requirements

**Data Requirements:**
- `data/amazon_sales.csv` must exist

**Docker Compose Files:**
- `docker-compose.yml` (main app)
- `docker-compose.monitoring.yml` (monitoring stack)

---

## 📊 After Deployment

**Service URLs:**
- API: http://localhost:8000
- UI: http://localhost:8000/ui
- API Docs: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Metrics: http://localhost:8001/metrics

**Useful Commands:**
```bash
# View logs
docker compose logs -f

# Check containers
docker ps

# Restart app
docker compose restart

# Complete teardown
scripts/stop.sh
docker compose down -v
docker compose -f docker-compose.monitoring.yml down -v
```

---

## 🐛 Troubleshooting

### "Docker is not running"
**Fix:** Start Docker Desktop, wait for it to fully start

### "data/amazon_sales.csv not found"
**Fix:** Ensure data file exists at `data/amazon_sales.csv`

### Port already in use
**Fix:**
```bash
lsof -ti:8000 | xargs kill -9  # Kill process on port 8000
```

### Tests failing
**Fix:**
```bash
scripts/test.sh  # Run tests independently
cat test_results.txt  # View detailed results
```

### Container unhealthy
**Fix:**
```bash
docker compose logs  # Check logs
docker compose down && scripts/deploy.sh  # Rebuild
```

---

## 📖 Full Documentation

See `DEPLOYMENT.md` in project root for complete documentation.
