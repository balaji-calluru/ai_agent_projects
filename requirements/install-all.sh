#!/bin/bash
# ============================================================================
# Install All Dependencies (Use with Caution - May Have Conflicts)
# ============================================================================
# This script installs all project dependencies.
# WARNING: Some projects have conflicting dependencies.
# Consider using separate virtual environments instead.

set -e  # Exit on error

echo "============================================================================"
echo "Installing All Project Dependencies"
echo "============================================================================"
echo ""
echo "⚠️  WARNING: This may cause dependency conflicts!"
echo "   Projects 07 and 10 (CrewAI) conflict with Projects 01 and 09."
echo "   Consider using separate virtual environments instead."
echo ""
read -p "Continue anyway? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 1
fi

echo ""
echo "Step 1: Installing base requirements..."
pip install -r requirements/base.txt

echo ""
echo "Step 2: Installing project-specific requirements..."
pip install -r requirements/01-rag-pipeline.txt || echo "⚠️  Warning: Project 01 installation had issues"
pip install -r requirements/02-langgraph.txt || echo "⚠️  Warning: Project 02 installation had issues"
pip install -r requirements/03-rag-langchain.txt || echo "⚠️  Warning: Project 03 installation had issues"
pip install -r requirements/04-research-agent.txt || echo "⚠️  Warning: Project 04 installation had issues"
pip install -r requirements/05-gemini-api.txt || echo "⚠️  Warning: Project 05 installation had issues"
pip install -r requirements/06-game-agent.txt || echo "⚠️  Warning: Project 06 installation had issues"
pip install -r requirements/07-crewai.txt || echo "⚠️  Warning: Project 07 installation had issues (expected conflicts)"
pip install -r requirements/08-openai-api.txt || echo "⚠️  Warning: Project 08 installation had issues"
pip install -r requirements/09-agentic-ai.txt || echo "⚠️  Warning: Project 09 installation had issues (expected conflicts)"
pip install -r requirements/10-image-research.txt || echo "⚠️  Warning: Project 10 installation had issues (expected conflicts)"

echo ""
echo "============================================================================"
echo "Installation Complete"
echo "============================================================================"
echo ""
echo "If you encountered conflicts, use separate virtual environments:"
echo "  python -m venv venv-projectXX"
echo "  source venv-projectXX/bin/activate"
echo "  pip install -r requirements/base.txt -r requirements/XX-project-name.txt"
echo ""

