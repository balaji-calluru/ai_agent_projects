# Practicing AI Agent Projects

- [10 AI Agent Projects to Build This Weekend](https://amanxai.com/2026/01/03/10-ai-agent-projects-to-build-this-weekend/)

## 🚀 Quick Start

### Automated Setup (Recommended)

**Setup all projects with virtual environments and Jupyter kernels:**
```bash
./setup-all-projects.sh
```

**Setup specific projects:**
```bash
./setup-all-projects.sh 10              # Only project 10
./setup-all-projects.sh 07 10           # Projects 07 and 10
```

**Setup single project:**
```bash
./setup-project.sh 10
```

This automatically:
- ✅ Creates isolated virtual environments
- ✅ Installs all dependencies
- ✅ Registers Jupyter kernels
- ✅ No dependency conflicts!

### Manual Installation

**For a specific project:**
```bash
pip install -r requirements/base.txt -r requirements/XX-project-name.txt
```

**Example (Project 10 - Image Research Agent):**
```bash
pip install -r requirements/base.txt -r requirements/10-image-research.txt
```

**Using virtual environments (manual):**
```bash
python -m venv venv-project10
source venv-project10/bin/activate  # Windows: venv-project10\Scripts\activate
pip install -r requirements/base.txt -r requirements/10-image-research.txt
```

### 📚 Documentation

- **Setup Guide**: [`SETUP_GUIDE.md`](SETUP_GUIDE.md) - Virtual environments and kernels
- **Scripts Guide**: [`SCRIPTS_README.md`](SCRIPTS_README.md) - Setup script documentation
- **Quick Start**: [`requirements/QUICK_START.md`](requirements/QUICK_START.md) - Requirements installation
- **Full Guide**: [`requirements/README.md`](requirements/README.md) - Requirements details
- **Migration Guide**: [`REQUIREMENTS_MIGRATION.md`](REQUIREMENTS_MIGRATION.md) - From old structure

### ⚠️ Important Notes

- **Old `requirements.txt` is deprecated** - Use modular files in `requirements/` directory
- **CrewAI projects (07, 10)** conflict with other projects - Use separate virtual environments
- **See conflict groups** in `requirements/README.md`

## 📁 Project Structure

```
src/
├── 01 - Building an Agentic RAG Pipeline.ipynb
├── 02 - Build a Multi-Agent System With LangGraph.ipynb
├── 03 - Build a Real-Time AI Assistant Using RAG + LangChain.ipynb
├── 04 - Build an AI Agent to Automate Your Research.ipynb
├── 05 - Building a Multi-Agent System using Gemini API.ipynb
├── 06 - Build an AI Agent to Master a Game using Python.ipynb
├── 07 - Building AI Agents with CrewAI using Python.ipynb
├── 08 - Building an AI Agent using OpenAI API.ipynb
├── 09 - Building an AI Agent using Agentic AI.ipynb
└── 10 - Build an AI research agent for image analysis.ipynb

requirements/
├── base.txt                    # Common dependencies
├── 01-rag-pipeline.txt        # Project-specific requirements
├── 02-langgraph.txt
├── ... (one file per project)
└── README.md                   # Detailed documentation
```
