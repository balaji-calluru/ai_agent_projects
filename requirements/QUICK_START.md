# Quick Start Guide - Requirements Installation

## 🚀 Fastest Way: Install for One Project

```bash
# Example: Install dependencies for Project 10 (Image Research Agent)
pip install -r requirements/base.txt -r requirements/10-image-research.txt
```

## 📦 Recommended: Use Virtual Environments

### For CrewAI Projects (07, 10)

```bash
# Create virtual environment
python -m venv venv-crewai
source venv-crewai/bin/activate  # On Windows: venv-crewai\Scripts\activate

# Install dependencies
pip install -r requirements/base.txt -r requirements/10-image-research.txt
```

### For LangChain Projects (01-06)

```bash
# Create virtual environment
python -m venv venv-langchain
source venv-langchain/bin/activate

# Install dependencies (example for Project 01)
pip install -r requirements/base.txt -r requirements/01-rag-pipeline.txt
```

### For ML Projects (09)

```bash
# Create virtual environment
python -m venv venv-ml
source venv-ml/bin/activate

# Install dependencies
pip install -r requirements/base.txt -r requirements/09-agentic-ai.txt
```

## 🔍 Check What You Need

| Project | File | Key Dependencies |
|---------|------|------------------|
| 01 | `01-rag-pipeline.txt` | langchain-chroma, chromadb>=1.4.0 |
| 02 | `02-langgraph.txt` | langgraph |
| 03 | `03-rag-langchain.txt` | langchain-ollama, duckduckgo-search |
| 04 | `04-research-agent.txt` | duckduckgo-search |
| 05 | `05-gemini-api.txt` | google-genai, langgraph |
| 06 | `06-game-agent.txt` | gymnasium, pygame |
| 07 | `07-crewai.txt` | crewai, crewai-tools |
| 08 | `08-openai-api.txt` | openai |
| 09 | `09-agentic-ai.txt` | torch, transformers, yfinance |
| 10 | `10-image-research.txt` | crewai, crewai-tools, pillow, pydantic |

## ⚠️ Known Conflicts

- **Projects 07 & 10** (CrewAI) conflict with **Projects 01 & 09**
- Use separate virtual environments for conflicting projects

## 📚 Full Documentation

See `requirements/README.md` for detailed information.

