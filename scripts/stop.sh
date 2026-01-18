#!/bin/bash

# Marketing AI - Stop All Services Script
# This script stops all running containers

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Marketing AI - Stopping All Services"
echo "=========================================="

# Stop main application
echo ""
echo "Stopping main application..."
docker compose down

# Stop monitoring stack
echo ""
echo "Stopping monitoring stack..."
docker compose -f docker-compose.monitoring.yml down

echo ""
echo "=========================================="
echo "All services stopped!"
echo "=========================================="
echo ""
echo "To restart, run: scripts/deploy.sh"
echo ""
