#!/bin/bash

# Marketing AI - Complete Deployment Script
# Automates: Training → Testing → Server Deployment → Monitoring Deployment

set -e  # Exit on any error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Marketing AI - Complete Deployment"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if Docker is running
echo "Step 1: Checking prerequisites..."
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi
print_status "Docker is running"

# Check if data file exists
if [ ! -f "data/amazon_sales.csv" ]; then
    print_error "data/amazon_sales.csv not found!"
    exit 1
fi
print_status "Data file found"
echo ""

# Step 2: Train Model
echo "=========================================="
echo "Step 2: Training Model"
echo "=========================================="
bash scripts/train.sh
echo ""

# Step 3: Run Tests (non-blocking)
echo "=========================================="
echo "Step 3: Running Unit Tests"
echo "=========================================="
if bash scripts/test.sh; then
    print_status "All tests passed!"
else
    print_warning "Some tests failed, but continuing deployment..."
    echo "  Check test_results.txt for details"
fi
echo ""

# Step 4: Deploy Main Server
echo "=========================================="
echo "Step 4: Deploying Main Server"
echo "=========================================="

# Stop existing containers
print_status "Stopping existing containers..."
docker compose down 2>/dev/null || true

# Build and start main server
print_status "Building Docker image..."
docker compose build

print_status "Starting main server container..."
docker compose up -d

# Wait for service to be ready
echo ""
echo "Waiting for service to be ready..."
sleep 5

max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_status "Main server is healthy!"
        break
    fi
    attempt=$((attempt + 1))
    echo "  Attempt $attempt/$max_attempts..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    print_error "Main server did not become healthy"
    echo "Check logs with: docker-compose logs -f"
    exit 1
fi
echo ""

# Step 5: Deploy Monitoring (Prometheus + Grafana)
echo "=========================================="
echo "Step 5: Deploying Monitoring Stack"
echo "=========================================="

# Stop existing monitoring containers
print_status "Stopping existing monitoring containers..."
docker compose -f docker-compose.monitoring.yml down 2>/dev/null || true

# Start monitoring stack
print_status "Starting Prometheus and Grafana..."
docker compose -f docker-compose.monitoring.yml up -d

# Wait for Prometheus
echo ""
echo "Waiting for Prometheus to be ready..."
sleep 5

max_attempts=20
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
        print_status "Prometheus is healthy!"
        break
    fi
    attempt=$((attempt + 1))
    echo "  Attempt $attempt/$max_attempts..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    print_warning "Prometheus did not become healthy within timeout"
fi

# Wait for Grafana
echo ""
echo "Waiting for Grafana to be ready..."
sleep 5

max_attempts=20
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
        print_status "Grafana is healthy!"
        break
    fi
    attempt=$((attempt + 1))
    echo "  Attempt $attempt/$max_attempts..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    print_warning "Grafana did not become healthy within timeout"
fi

echo ""
echo "=========================================="
echo "Deployment Complete! 🚀"
echo "=========================================="
echo ""
echo "Service URLs:"
echo "  📊 Main Application:"
echo "     - API:        http://localhost:8000"
echo "     - UI:         http://localhost:8000/ui"
echo "     - API Docs:   http://localhost:8000/docs"
echo "     - Health:     http://localhost:8000/health"
echo ""
echo "  📈 Monitoring:"
echo "     - Prometheus: http://localhost:9090"
echo "     - Grafana:    http://localhost:3000 (admin/admin)"
echo "     - Metrics:    http://localhost:8001/metrics"
echo ""
echo "Running Containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(NAMES|marketing-ai|prometheus|grafana)"
echo ""
echo "Useful Commands:"
echo "  - View app logs:        docker-compose logs -f"
echo "  - View monitoring logs: docker-compose -f docker-compose.monitoring.yml logs -f"
echo "  - Stop all:             docker-compose down && docker-compose -f docker-compose.monitoring.yml down"
echo "  - Restart app:          docker-compose restart"
echo ""


# """
# http://localhost:8000/ui/index.html
# """