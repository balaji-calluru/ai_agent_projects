# Setup Guide - Virtual Environments and Jupyter Kernels

This guide explains how to set up virtual environments and Jupyter kernels for all projects.

## 🚀 Quick Start

### Setup All Projects (Recommended)

```bash
./setup-all-projects.sh
```

This will:
1. Create virtual environments for all 10 projects
2. Install dependencies for each project
3. Register Jupyter kernels for each project

### Setup Specific Projects

```bash
# Setup only project 10
./setup-all-projects.sh 10

# Setup projects 07 and 10
./setup-all-projects.sh 07 10
```

### Setup Single Project (Alternative)

```bash
# Setup project 10
./setup-project.sh 10

# Force recreate (if already exists)
./setup-project.sh 10 --force
```

## 📋 What Gets Created

For each project, the script creates:

1. **Virtual Environment**: `venv-projectXX/`
   - Isolated Python environment
   - Contains only project-specific dependencies
   - No conflicts between projects

2. **Jupyter Kernel**: `projectXX`
   - Registered with Jupyter
   - Display name: "Project XX: Project Name"
   - Can be selected in Jupyter notebook kernel dropdown

## 🎯 Using the Setup

### Activate Virtual Environment

```bash
# For project 10
source venv-project10/bin/activate

# Install additional packages (if needed)
pip install some-package

# Deactivate when done
deactivate
```

### Use Jupyter Kernel

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

2. **Select Kernel**:
   - Open your notebook
   - Click on kernel name (top right)
   - Select "Project XX: Project Name"
   - Kernel will use the project's virtual environment

### List All Kernels

```bash
# Using the management script
./manage-kernels.sh list

# Or directly
jupyter kernelspec list
```

## 🔧 Management Scripts

### List Kernels

```bash
./manage-kernels.sh list
```

### Remove a Kernel

```bash
./manage-kernels.sh remove 10
```

### Reinstall a Kernel

```bash
./manage-kernels.sh reinstall 10
```

## 📁 Directory Structure After Setup

```
ai_agent_projects/
├── venv-project01/          # Virtual environment for project 01
├── venv-project02/          # Virtual environment for project 02
├── venv-project03/          # Virtual environment for project 03
├── venv-project04/          # Virtual environment for project 04
├── venv-project05/          # Virtual environment for project 05
├── venv-project06/          # Virtual environment for project 06
├── venv-project07/          # Virtual environment for project 07
├── venv-project08/          # Virtual environment for project 08
├── venv-project09/          # Virtual environment for project 09
├── venv-project10/          # Virtual environment for project 10
├── requirements/            # Requirements files
│   ├── base.txt
│   ├── 01-rag-pipeline.txt
│   └── ...
├── setup-all-projects.sh    # Main setup script
├── setup-project.sh         # Single project setup
└── manage-kernels.sh        # Kernel management
```

## ⚠️ Important Notes

### Virtual Environments

- Each project has its own isolated environment
- No dependency conflicts between projects
- Can have different Python package versions
- Environments are stored in `venv-projectXX/` directories

### Jupyter Kernels

- Kernels are registered in your user directory
- Each kernel points to its project's virtual environment
- You can use any kernel with any notebook
- Kernels persist across Jupyter sessions

### Cleanup

To remove a project's setup:

```bash
# Remove virtual environment
rm -rf venv-project10

# Remove Jupyter kernel
./manage-kernels.sh remove 10
```

## 🔍 Troubleshooting

### Kernel Not Appearing in Jupyter

1. **Check if kernel is registered**:
   ```bash
   jupyter kernelspec list
   ```

2. **Reinstall kernel**:
   ```bash
   ./manage-kernels.sh reinstall 10
   ```

3. **Restart Jupyter**:
   - Close Jupyter completely
   - Restart: `jupyter notebook`

### Virtual Environment Issues

1. **Recreate environment**:
   ```bash
   ./setup-project.sh 10 --force
   ```

2. **Check Python version**:
   ```bash
   source venv-project10/bin/activate
   python --version
   ```

### Installation Errors

1. **Check requirements file exists**:
   ```bash
   ls requirements/10-image-research.txt
   ```

2. **Install manually**:
   ```bash
   source venv-project10/bin/activate
   pip install -r requirements/base.txt
   pip install -r requirements/10-image-research.txt
   ```

## 📊 Project Status Check

Check which projects are set up:

```bash
# List all virtual environments
ls -d venv-project* 2>/dev/null | wc -l

# List all kernels
./manage-kernels.sh list
```

## 🎓 Best Practices

1. **Use separate environments** - Each project has its own environment
2. **Select correct kernel** - Always use the project's kernel in Jupyter
3. **Keep environments updated** - Reinstall if you modify requirements
4. **Clean up unused projects** - Remove environments you're not using

## 📚 Related Documentation

- **Requirements Guide**: `requirements/README.md`
- **Quick Start**: `requirements/QUICK_START.md`
- **Migration Guide**: `REQUIREMENTS_MIGRATION.md`

