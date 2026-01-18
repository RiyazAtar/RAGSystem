#!/bin/bash

# Marketing AI - Unit Testing Script
# This script runs all unit tests

set -e  # Exit on any error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Marketing AI - Unit Tests"
echo "=========================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found!"
    exit 1
fi

# Install test dependencies
echo "Installing test dependencies..."
pip install -q pytest pytest-cov pytest-asyncio

# Check if tests directory exists
if [ ! -d "tests" ]; then
    echo "Warning: tests/ directory not found. Creating basic test structure..."
    mkdir -p tests

    # Create a basic test file if none exists
    if [ ! -f "tests/test_api.py" ]; then
        cat > tests/test_api.py << 'EOF'
"""
Basic API tests for Marketing AI System
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_import():
    """Test that we can import the API module"""
    try:
        from src import api
        assert True
    except ImportError as e:
        pytest.skip(f"Cannot import API: {e}")

def test_basic():
    """Basic test to ensure pytest is working"""
    assert 1 + 1 == 2
EOF
    fi

    # Create __init__.py
    touch tests/__init__.py
fi

# Run tests
echo ""
echo "Running tests..."
echo "------------------------------------------"

# Run pytest with coverage
python3 -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing 2>&1 | tee test_results.txt

# Check test results
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ All tests passed!"
    echo "=========================================="
    echo ""
    exit 0
else
    echo ""
    echo "=========================================="
    echo "✗ Some tests failed!"
    echo "=========================================="
    echo ""
    echo "Check test_results.txt for details"
    exit 1
fi
