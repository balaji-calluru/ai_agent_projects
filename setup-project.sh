#!/bin/bash
# ============================================================================
# Setup Single Project - Virtual Environment and Jupyter Kernel
# ============================================================================
# Quick script to setup a single project
#
# Usage:
#   ./setup-project.sh 10              # Setup project 10
#   ./setup-project.sh 10 --force      # Force recreate even if exists
#
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_NUM=$1
FORCE=${2:-""}

# Project mapping
declare -A PROJECTS=(
    ["01"]="01-rag-pipeline"
    ["02"]="02-langgraph"
    ["03"]="03-rag-langchain"
    ["04"]="04-research-agent"
    ["05"]="05-gemini-api"
    ["06"]="06-game-agent"
    ["07"]="07-crewai"
    ["08"]="08-openai-api"
    ["09"]="09-agentic-ai"
    ["10"]="10-image-research"
)

declare -A PROJECT_NAMES=(
    ["01"]="Building an Agentic RAG Pipeline"
    ["02"]="Build a Multi-Agent System With LangGraph"
    ["03"]="Build a Real-Time AI Assistant Using RAG + LangChain"
    ["04"]="Build an AI Agent to Automate Your Research"
    ["05"]="Building a Multi-Agent System using Gemini API"
    ["06"]="Build an AI Agent to Master a Game using Python"
    ["07"]="Building AI Agents with CrewAI using Python"
    ["08"]="Building an AI Agent using OpenAI API"
    ["09"]="Building an AI Agent using Agentic AI"
    ["10"]="Build an AI research agent for image analysis"
)

if [ -z "$PROJECT_NUM" ]; then
    echo -e "${RED}❌ Error: Project number required${NC}"
    echo "Usage: ./setup-project.sh <project_number> [--force]"
    echo "Example: ./setup-project.sh 10"
    exit 1
fi

if [ -z "${PROJECTS[$PROJECT_NUM]}" ]; then
    echo -e "${RED}❌ Error: Invalid project number: ${PROJECT_NUM}${NC}"
    echo "Valid project numbers: 01-10"
    exit 1
fi

REQ_FILE="${PROJECTS[$PROJECT_NUM]}"
PROJECT_NAME="${PROJECT_NAMES[$PROJECT_NUM]}"
VENV_NAME="venv-project${PROJECT_NUM}"
KERNEL_NAME="project${PROJECT_NUM}"
DISPLAY_NAME="Project ${PROJECT_NUM}: ${PROJECT_NAME}"

echo -e "${BLUE}Setting up Project ${PROJECT_NUM}: ${PROJECT_NAME}${NC}"
echo ""

# Check if requirements file exists
if [ ! -f "requirements/${REQ_FILE}.txt" ]; then
    echo -e "${RED}❌ Requirements file not found: requirements/${REQ_FILE}.txt${NC}"
    exit 1
fi

# Create or recreate virtual environment
if [ -d "${VENV_NAME}" ]; then
    if [ "$FORCE" == "--force" ]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf "${VENV_NAME}"
    else
        echo -e "${YELLOW}⚠️  Virtual environment '${VENV_NAME}' already exists.${NC}"
        echo "Use --force to recreate it."
        exit 0
    fi
fi

echo -e "${BLUE}Creating virtual environment...${NC}"
python3 -m venv "${VENV_NAME}"

echo -e "${BLUE}Activating environment and upgrading pip...${NC}"
source "${VENV_NAME}/bin/activate"
pip install --upgrade pip --quiet

echo -e "${BLUE}Installing base requirements...${NC}"
pip install -q -r requirements/base.txt

echo -e "${BLUE}Installing project-specific requirements...${NC}"
pip install -q -r "requirements/${REQ_FILE}.txt"

echo -e "${BLUE}Registering Jupyter kernel...${NC}"
# Remove existing kernel if it exists
jupyter kernelspec remove -f "${KERNEL_NAME}" 2>/dev/null || true
python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "${DISPLAY_NAME}"

deactivate

echo ""
echo -e "${GREEN}✅ Project ${PROJECT_NUM} setup complete!${NC}"
echo ""
echo "To activate the environment:"
echo "  source ${VENV_NAME}/bin/activate"
echo ""
echo "To use in Jupyter:"
echo "  Select kernel '${DISPLAY_NAME}' from the kernel dropdown"

