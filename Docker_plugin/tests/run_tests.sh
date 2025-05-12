#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d ".venv"]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install test dependencies
pip install pytest pytest-asyncio pytest-cov websockets

# Run tests with coverage
pytest tests/ --cov=src/ --cov-report=html -v

# Deactivate virtual environment
deactivate