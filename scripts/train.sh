#!/bin/bash

# Marketing AI - Model Training Script
# This script trains the discount prediction model

set -e  # Exit on any error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Marketing AI - Model Training"
echo "=========================================="

# Check if data file exists
if [ ! -f "data/amazon_sales.csv" ]; then
    echo "Error: data/amazon_sales.csv not found!"
    exit 1
fi

# Create models directory if it doesn't exist
mkdir -p models

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

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
fi

# Run training
echo ""
echo "Starting model training..."
echo "------------------------------------------"
python3 scripts/train.py

echo ""
echo "=========================================="
echo "✓ Training completed successfully!"
echo "=========================================="
echo ""
echo "Model artifacts saved in: models/"
echo "  - discount_model.joblib"
echo "  - processor.joblib"
echo "  - rag_system/"
echo ""
