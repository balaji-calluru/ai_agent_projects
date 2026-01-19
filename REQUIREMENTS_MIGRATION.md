# Requirements Migration Guide

## ✅ What Changed

The monolithic `requirements.txt` has been **refactored into a modular structure** to solve dependency conflicts and speed up installation.

## 📁 New Structure

```
requirements/
├── base.txt                    # Common dependencies (minimal, safe)
├── 01-rag-pipeline.txt        # Project 01 specific
├── 02-langgraph.txt           # Project 02 specific
├── 03-rag-langchain.txt       # Project 03 specific
├── 04-research-agent.txt      # Project 04 specific
├── 05-gemini-api.txt          # Project 05 specific
├── 06-game-agent.txt          # Project 06 specific
├── 07-crewai.txt              # Project 07 specific
├── 08-openai-api.txt          # Project 08 specific
├── 09-agentic-ai.txt          # Project 09 specific
├── 10-image-research.txt      # Project 10 specific
├── README.md                  # Detailed documentation
├── QUICK_START.md             # Quick reference guide
└── install-all.sh             # Script to install all (use with caution)
```

## 🚀 How to Use

### For a Single Project (Recommended)

```bash
# Install base + project-specific dependencies
pip install -r requirements/base.txt -r requirements/10-image-research.txt
```

### Using Virtual Environments (Best Practice)

```bash
# Create environment for Project 10
python -m venv venv-project10
source venv-project10/bin/activate  # Windows: venv-project10\Scripts\activate

# Install only what you need
pip install -r requirements/base.txt -r requirements/10-image-research.txt
```

## 🎯 Benefits

1. ✅ **No More Conflicts** - Each project has compatible dependencies
2. ✅ **Faster Installation** - Install only what you need
3. ✅ **Clear Dependencies** - See exactly what each project needs
4. ✅ **Easy Maintenance** - Update one project without affecting others

## ⚠️ Important Notes

### Old `requirements.txt`

The old `requirements.txt` is now **deprecated** and contains a warning message. It's kept for reference but should not be used for installation.

### Conflict Groups

- **Group 1 (Compatible)**: Projects 01-06, 08 - Can be installed together
- **Group 2 (CrewAI)**: Projects 07, 10 - Need separate environment
- **Group 3 (ML)**: Project 09 - Can conflict with CrewAI projects

### Migration Steps

1. **Remove old installations** (optional, if you want a clean start):
   ```bash
   pip uninstall -y $(pip freeze | cut -d= -f1)
   ```

2. **Install base requirements**:
   ```bash
   pip install -r requirements/base.txt
   ```

3. **Install project-specific requirements**:
   ```bash
   pip install -r requirements/XX-project-name.txt
   ```

## 📊 Project Dependency Summary

| Project | Base | Specific | Conflicts With |
|---------|------|----------|----------------|
| 01 | ✅ | langchain-chroma, chromadb | CrewAI projects |
| 02 | ✅ | langgraph | None |
| 03 | ✅ | langchain-ollama, duckduckgo | None |
| 04 | ✅ | duckduckgo-search | None |
| 05 | ✅ | google-genai, langgraph | None |
| 06 | ✅ | gymnasium, pygame | None |
| 07 | ✅ | crewai, crewai-tools | 01, 09 |
| 08 | ✅ | openai | None |
| 09 | ✅ | torch, transformers | CrewAI projects |
| 10 | ✅ | crewai, crewai-tools, pillow, pydantic | 01, 09 |

## 🔧 Troubleshooting

### If you get conflicts:

1. **Use separate virtual environments** (recommended)
2. **Check the project's requirements file** for specific versions
3. **See `requirements/README.md`** for detailed conflict resolution

### Quick Test

```bash
# Test installation for Project 10
pip install --dry-run -r requirements/base.txt -r requirements/10-image-research.txt
```

If this succeeds, the actual installation will work.

## 📚 Documentation

- **Quick Start**: `requirements/QUICK_START.md`
- **Full Guide**: `requirements/README.md`
- **Installation Script**: `requirements/install-all.sh` (use with caution)

