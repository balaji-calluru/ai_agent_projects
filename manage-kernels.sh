#!/bin/bash
# ============================================================================
# Manage Jupyter Kernels
# ============================================================================
# Utility script to list, remove, or reinstall Jupyter kernels
#
# Usage:
#   ./manage-kernels.sh list              # List all kernels
#   ./manage-kernels.sh remove 10         # Remove kernel for project 10
#   ./manage-kernels.sh reinstall 10      # Reinstall kernel for project 10
#
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ACTION=$1
PROJECT_NUM=$2

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

list_kernels() {
    echo -e "${BLUE}Registered Jupyter Kernels:${NC}"
    echo ""
    jupyter kernelspec list
    echo ""
    
    # Check for project kernels
    echo -e "${BLUE}Project-specific kernels:${NC}"
    for i in {01..10}; do
        kernel_name="project${i}"
        if jupyter kernelspec list | grep -q "${kernel_name}"; then
            project_name="${PROJECT_NAMES[$i]}"
            echo -e "  ${GREEN}✅${NC} project${i} - Project ${i}: ${project_name}"
        else
            echo -e "  ${YELLOW}⚠️${NC}  project${i} - Not registered"
        fi
    done
}

remove_kernel() {
    if [ -z "$PROJECT_NUM" ]; then
        echo -e "${RED}❌ Error: Project number required${NC}"
        echo "Usage: ./manage-kernels.sh remove <project_number>"
        exit 1
    fi
    
    kernel_name="project${PROJECT_NUM}"
    project_name="${PROJECT_NAMES[$PROJECT_NUM]}"
    
    if jupyter kernelspec list | grep -q "${kernel_name}"; then
        echo -e "${BLUE}Removing kernel '${kernel_name}' (Project ${PROJECT_NUM}: ${project_name})...${NC}"
        jupyter kernelspec remove -f "${kernel_name}"
        echo -e "${GREEN}✅ Kernel removed${NC}"
    else
        echo -e "${YELLOW}⚠️  Kernel '${kernel_name}' not found${NC}"
    fi
}

reinstall_kernel() {
    if [ -z "$PROJECT_NUM" ]; then
        echo -e "${RED}❌ Error: Project number required${NC}"
        echo "Usage: ./manage-kernels.sh reinstall <project_number>"
        exit 1
    fi
    
    VENV_NAME="venv-project${PROJECT_NUM}"
    KERNEL_NAME="project${PROJECT_NUM}"
    PROJECT_NAME="${PROJECT_NAMES[$PROJECT_NUM]}"
    DISPLAY_NAME="Project ${PROJECT_NUM}: ${PROJECT_NAME}"
    
    if [ ! -d "${VENV_NAME}" ]; then
        echo -e "${RED}❌ Virtual environment '${VENV_NAME}' not found${NC}"
        echo "Run ./setup-project.sh ${PROJECT_NUM} first"
        exit 1
    fi
    
    echo -e "${BLUE}Reinstalling kernel for Project ${PROJECT_NUM}...${NC}"
    
    # Remove existing kernel
    jupyter kernelspec remove -f "${KERNEL_NAME}" 2>/dev/null || true
    
    # Activate environment and install kernel
    source "${VENV_NAME}/bin/activate"
    python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "${DISPLAY_NAME}"
    deactivate
    
    echo -e "${GREEN}✅ Kernel reinstalled${NC}"
}

case "$ACTION" in
    list)
        list_kernels
        ;;
    remove)
        remove_kernel
        ;;
    reinstall)
        reinstall_kernel
        ;;
    *)
        echo -e "${RED}❌ Error: Invalid action '${ACTION}'${NC}"
        echo ""
        echo "Usage:"
        echo "  ./manage-kernels.sh list                    # List all kernels"
        echo "  ./manage-kernels.sh remove <project_num>    # Remove kernel"
        echo "  ./manage-kernels.sh reinstall <project_num> # Reinstall kernel"
        echo ""
        echo "Examples:"
        echo "  ./manage-kernels.sh list"
        echo "  ./manage-kernels.sh remove 10"
        echo "  ./manage-kernels.sh reinstall 10"
        exit 1
        ;;
esac

