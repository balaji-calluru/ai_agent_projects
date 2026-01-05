#!/bin/bash
# Script to launch Jupyter Notebook for viewing the notebook

cd "$(dirname "$0")"
source venv/bin/activate
echo "Starting Jupyter Notebook..."
echo "The notebook will open in your browser."
echo "Navigate to: src/Build a Multi-Agent System With LangGraph.ipynb"
echo ""
jupyter notebook

