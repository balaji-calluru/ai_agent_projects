#!/bin/bash
# Script to launch Jupyter Notebook for viewing the notebooks
# This script should be run from the project root or from the tools directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of tools directory)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
else
    echo "⚠️  Warning: Virtual environment not found at venv/bin/activate"
    echo "   Continuing without activation..."
fi

echo "Starting Jupyter Notebook..."
echo "Project root: $PROJECT_ROOT"
echo "The notebook will open in your browser."
echo "Navigate to: src/ to find your notebooks"
echo ""
jupyter notebook

