# Installation Notes - Dependency Conflicts Resolution

## ChromaDB and CrewAI Conflict

### Problem
There is a known dependency conflict between:
- **CrewAI** requires `chromadb~=1.1.0` (version 1.1.x only)
- **langchain-chroma** requires `chromadb>=1.3.5`

These requirements are incompatible and cannot be satisfied simultaneously.

### Solution Applied

1. **Removed explicit chromadb requirement** - Let CrewAI install its required version automatically
2. **Commented out langchain-chroma** - Made it optional to avoid conflicts
3. **Added clear documentation** - Explained the conflict and workarounds

### Installation Instructions

#### For Notebook 10 (Image Research Agent with CrewAI):

```bash
# Install all dependencies (CrewAI will install chromadb~=1.1.0 automatically)
pip install -r requirements.txt
```

#### If you need langchain-chroma separately:

```bash
# Option 1: Install in a separate virtual environment
python -m venv venv_langchain
source venv_langchain/bin/activate  # On Windows: venv_langchain\Scripts\activate
pip install langchain-chroma chromadb>=1.3.5

# Option 2: Install without dependencies (not recommended)
pip install langchain-chroma --no-deps
# Then manually manage chromadb version conflicts
```

### Packages Installed by CrewAI

When you install `crewai>=1.7.0`, it will automatically install:
- `chromadb~=1.1.0` (this is the conflict source)
- Other dependencies as needed

### Verification

After installation, verify CrewAI is working:

```python
from crewai import Agent, Task, Crew
print("✅ CrewAI installed successfully")
```

### Notes

- **Notebook 10** (Image Research Agent) uses CrewAI and will work with the current setup
- If you need both CrewAI and langchain-chroma, use separate virtual environments
- The conflict is a known issue in the CrewAI ecosystem and may be resolved in future versions

## Other Dependency Notes

### Tokenizers Conflict
- CrewAI requires `tokenizers~=0.20.3`
- Transformers requires `tokenizers>=0.22.0`
- Current solution: Using `tokenizers>=0.22.0,<=0.23.0` which works for most cases

### Installation Order
If you encounter issues, install in this order:
1. Core dependencies (langchain, etc.)
2. CrewAI and its tools
3. Other packages

