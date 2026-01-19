# Setup Scripts Documentation

## 📜 Available Scripts

### 1. `setup-all-projects.sh` - Main Setup Script

**Purpose**: Creates virtual environments and Jupyter kernels for all or selected projects.

**Usage**:
```bash
# Setup all projects
./setup-all-projects.sh

# Setup specific projects
./setup-all-projects.sh 10
./setup-all-projects.sh 07 10
./setup-all-projects.sh 01 02 03
```

**What it does**:
1. Creates virtual environment `venv-projectXX/` for each project
2. Activates the environment
3. Upgrades pip
4. Installs base requirements (`requirements/base.txt`)
5. Installs project-specific requirements (`requirements/XX-project-name.txt`)
6. Registers Jupyter kernel with display name "Project XX: Project Name"
7. Deactivates the environment

**Features**:
- ✅ Colored output for better readability
- ✅ Error handling and reporting
- ✅ Skips existing environments (with option to recreate)
- ✅ Summary report at the end
- ✅ Usage instructions

### 2. `setup-project.sh` - Single Project Setup

**Purpose**: Quick setup for a single project.

**Usage**:
```bash
# Setup project 10
./setup-project.sh 10

# Force recreate (if already exists)
./setup-project.sh 10 --force
```

**What it does**:
- Same as `setup-all-projects.sh` but for one project
- Simpler output
- Faster for single project setup

### 3. `manage-kernels.sh` - Kernel Management

**Purpose**: Manage Jupyter kernels (list, remove, reinstall).

**Usage**:
```bash
# List all kernels
./manage-kernels.sh list

# Remove kernel for project 10
./manage-kernels.sh remove 10

# Reinstall kernel for project 10
./manage-kernels.sh reinstall 10
```

**What it does**:
- Lists all registered Jupyter kernels
- Shows which project kernels are registered
- Removes kernels
- Reinstalls kernels (useful if kernel breaks)

## 🎯 Quick Examples

### Example 1: Setup All Projects

```bash
./setup-all-projects.sh
```

Output:
```
============================================================================
Setup All Projects - Virtual Environments and Jupyter Kernels
============================================================================

ℹ️  No project numbers specified. Setting up ALL projects...

============================================================================
ℹ️  Setting up Project 01: Building an Agentic RAG Pipeline
============================================================================
ℹ️  Step 1/4: Creating virtual environment 'venv-project01'...
✅ Virtual environment created
ℹ️  Step 2/4: Activating environment and upgrading pip...
✅ Pip upgraded
ℹ️  Step 3/4: Installing dependencies...
✅ Dependencies installed successfully
ℹ️  Step 4/4: Registering Jupyter kernel 'project01'...
✅ Jupyter kernel 'project01' registered
✅ Project 01 setup complete!

[... continues for all projects ...]

============================================================================
Setup Summary
============================================================================
✅ Successfully setup projects: 01 02 03 04 05 06 07 08 09 10
```

### Example 2: Setup Only CrewAI Projects

```bash
./setup-all-projects.sh 07 10
```

### Example 3: Check Kernel Status

```bash
./manage-kernels.sh list
```

Output:
```
Registered Jupyter Kernels:

  python3    /Users/username/Library/Jupyter/kernels/python3

Project-specific kernels:
  ✅ project01 - Project 01: Building an Agentic RAG Pipeline
  ✅ project02 - Project 02: Build a Multi-Agent System With LangGraph
  ⚠️   project03 - Not registered
  ...
```

## 🔍 Script Details

### Error Handling

- Scripts exit on error (`set -e`)
- Failed projects are tracked and reported
- Existing environments are detected and handled gracefully

### Output Formatting

- **ℹ️** Blue - Informational messages
- **✅** Green - Success messages
- **⚠️** Yellow - Warnings
- **❌** Red - Errors

### Virtual Environment Naming

- Format: `venv-projectXX/`
- Examples: `venv-project01/`, `venv-project10/`

### Jupyter Kernel Naming

- Kernel name: `projectXX`
- Display name: `Project XX: Project Name`
- Example: `project10` → "Project 10: Build an AI research agent for image analysis"

## 🛠️ Troubleshooting

### Script Won't Run

```bash
# Make sure scripts are executable
chmod +x setup-all-projects.sh
chmod +x setup-project.sh
chmod +x manage-kernels.sh
```

### Permission Denied

```bash
# Run with bash explicitly
bash setup-all-projects.sh
```

### Virtual Environment Already Exists

The script will ask if you want to recreate it. Or use:
```bash
./setup-project.sh 10 --force
```

### Kernel Not Working

1. Check if kernel is registered:
   ```bash
   ./manage-kernels.sh list
   ```

2. Reinstall kernel:
   ```bash
   ./manage-kernels.sh reinstall 10
   ```

3. Restart Jupyter

## 📊 Project Status

Check which projects are set up:

```bash
# Count virtual environments
ls -d venv-project* 2>/dev/null | wc -l

# List kernels
./manage-kernels.sh list
```

## 🎓 Best Practices

1. **Setup all at once**: Run `./setup-all-projects.sh` once to set up everything
2. **Use correct kernel**: Always select the project's kernel in Jupyter
3. **Keep environments updated**: Reinstall if requirements change
4. **Clean up unused**: Remove environments for projects you're not using

## 📚 Related Documentation

- **Setup Guide**: `SETUP_GUIDE.md`
- **Requirements Guide**: `requirements/README.md`
- **Quick Start**: `requirements/QUICK_START.md`

