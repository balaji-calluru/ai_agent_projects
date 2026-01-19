#!/bin/bash
# ============================================================================
# Setup All Projects - Virtual Environments and Jupyter Kernels
# ============================================================================
# This script creates virtual environments for each project, installs dependencies,
# and registers Jupyter kernels for easy notebook execution.
#
# Usage:
#   ./setup-all-projects.sh              # Setup all projects
#   ./setup-all-projects.sh 10           # Setup only project 10
#   ./setup-all-projects.sh 07 10        # Setup projects 07 and 10
#
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Project names for display
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

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to setup a single project
setup_project() {
    local project_num=$1
    local req_file=$2
    local project_name=$3
    local venv_name="venv-project${project_num}"
    
    echo ""
    echo "============================================================================"
    print_info "Setting up Project ${project_num}: ${project_name}"
    echo "============================================================================"
    
    # Check if requirements file exists
    if [ ! -f "requirements/${req_file}.txt" ]; then
        print_error "Requirements file not found: requirements/${req_file}.txt"
        return 1
    fi
    
    # Step 1: Create virtual environment
    print_info "Step 1/4: Creating virtual environment '${venv_name}'..."
    if [ -d "${venv_name}" ]; then
        print_warning "Virtual environment '${venv_name}' already exists."
        read -p "Remove and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "${venv_name}"
            python3 -m venv "${venv_name}"
            print_success "Virtual environment recreated"
        else
            print_info "Using existing virtual environment"
        fi
    else
        python3 -m venv "${venv_name}"
        print_success "Virtual environment created"
    fi
    
    # Step 2: Activate and upgrade pip
    print_info "Step 2/4: Activating environment and upgrading pip..."
    source "${venv_name}/bin/activate"
    pip install --upgrade pip --quiet
    print_success "Pip upgraded"
    
    # Step 3: Install dependencies
    print_info "Step 3/4: Installing dependencies..."
    print_info "  - Installing base requirements..."
    pip install -q -r requirements/base.txt
    
    print_info "  - Installing project-specific requirements..."
    if pip install -q -r "requirements/${req_file}.txt"; then
        print_success "Dependencies installed successfully"
    else
        print_error "Failed to install dependencies for project ${project_num}"
        deactivate
        return 1
    fi
    
    # Step 4: Create Jupyter kernel
    print_info "Step 4/4: Registering Jupyter kernel 'project${project_num}'..."
    kernel_name="project${project_num}"
    display_name="Project ${project_num}: ${project_name}"
    
    # Check if kernel already exists
    if python -m ipykernel install --user --name "${kernel_name}" --display-name "${display_name}" 2>/dev/null; then
        print_success "Jupyter kernel '${kernel_name}' registered"
    else
        # Try to remove existing kernel first
        jupyter kernelspec remove -f "${kernel_name}" 2>/dev/null || true
        python -m ipykernel install --user --name "${kernel_name}" --display-name "${display_name}"
        print_success "Jupyter kernel '${kernel_name}' registered"
    fi
    
    # Deactivate virtual environment
    deactivate
    
    print_success "Project ${project_num} setup complete!"
    echo ""
}

# Main execution
main() {
    echo "============================================================================"
    echo "Setup All Projects - Virtual Environments and Jupyter Kernels"
    echo "============================================================================"
    echo ""
    
    # Check if we're in the right directory
    if [ ! -d "requirements" ] || [ ! -f "requirements/base.txt" ]; then
        print_error "Please run this script from the project root directory"
        print_error "Expected: requirements/ directory and requirements/base.txt"
        exit 1
    fi
    
    # Determine which projects to setup
    if [ $# -eq 0 ]; then
        # Setup all projects
        print_info "No project numbers specified. Setting up ALL projects..."
        echo ""
        PROJECTS_TO_SETUP=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10")
    else
        # Setup specified projects
        PROJECTS_TO_SETUP=("$@")
        print_info "Setting up specified projects: ${PROJECTS_TO_SETUP[*]}"
        echo ""
    fi
    
    # Track results
    SUCCESSFUL=()
    FAILED=()
    
    # Setup each project
    for project_num in "${PROJECTS_TO_SETUP[@]}"; do
        if [ -z "${PROJECTS[$project_num]}" ]; then
            print_error "Invalid project number: ${project_num}"
            FAILED+=("${project_num}")
            continue
        fi
        
        req_file="${PROJECTS[$project_num]}"
        project_name="${PROJECT_NAMES[$project_num]}"
        
        if setup_project "${project_num}" "${req_file}" "${project_name}"; then
            SUCCESSFUL+=("${project_num}")
        else
            FAILED+=("${project_num}")
        fi
    done
    
    # Summary
    echo ""
    echo "============================================================================"
    echo "Setup Summary"
    echo "============================================================================"
    echo ""
    
    if [ ${#SUCCESSFUL[@]} -gt 0 ]; then
        print_success "Successfully setup projects: ${SUCCESSFUL[*]}"
        echo ""
        echo "Virtual environments created:"
        for project_num in "${SUCCESSFUL[@]}"; do
            echo "  - venv-project${project_num}/"
        done
        echo ""
        echo "Jupyter kernels registered:"
        for project_num in "${SUCCESSFUL[@]}"; do
            echo "  - project${project_num} (Project ${project_num})"
        done
        echo ""
    fi
    
    if [ ${#FAILED[@]} -gt 0 ]; then
        print_error "Failed to setup projects: ${FAILED[*]}"
        echo ""
    fi
    
    echo "============================================================================"
    echo "Usage Instructions"
    echo "============================================================================"
    echo ""
    echo "To use a project's virtual environment:"
    echo "  source venv-projectXX/bin/activate"
    echo ""
    echo "To use a project's Jupyter kernel:"
    echo "  1. Open Jupyter: jupyter notebook"
    echo "  2. Select 'Project XX: ...' from the kernel dropdown"
    echo ""
    echo "To list all kernels:"
    echo "  jupyter kernelspec list"
    echo ""
    echo "To remove a kernel:"
    echo "  jupyter kernelspec remove projectXX"
    echo ""
}

# Run main function
main "$@"

