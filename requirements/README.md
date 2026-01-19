# Requirements Management

This directory contains modular requirements files for each project in the workspace. This approach solves dependency conflicts and speeds up installation.

## Structure

- `base.txt` - Common dependencies shared across all projects
- `01-rag-pipeline.txt` through `10-image-research.txt` - Project-specific requirements

## Installation Methods

### Method 1: Install for a Specific Project (Recommended)

```bash
# Install base dependencies first
pip install -r requirements/base.txt

# Then install project-specific dependencies
pip install -r requirements/10-image-research.txt
```

Or in one command:
```bash
pip install -r requirements/base.txt -r requirements/10-image-research.txt
```

### Method 2: Install All Dependencies (Not Recommended)

```bash
# This may have conflicts - use separate virtual environments instead
pip install -r requirements/base.txt \
  -r requirements/01-rag-pipeline.txt \
  -r requirements/02-langgraph.txt \
  -r requirements/03-rag-langchain.txt \
  -r requirements/04-research-agent.txt \
  -r requirements/05-gemini-api.txt \
  -r requirements/06-game-agent.txt \
  -r requirements/07-crewai.txt \
  -r requirements/08-openai-api.txt \
  -r requirements/09-agentic-ai.txt \
  -r requirements/10-image-research.txt
```

### Method 3: Use Separate Virtual Environments (Best Practice)

```bash
# Create a virtual environment for each project
python -m venv venv-project10
source venv-project10/bin/activate  # On Windows: venv-project10\Scripts\activate

# Install only what you need
pip install -r requirements/base.txt -r requirements/10-image-research.txt
```

## Project-Specific Requirements

### Projects 01-06: LangChain-based (Compatible)
- Can be installed together
- Use `langchain-chroma` with `chromadb>=1.4.0`
- Use `tokenizers>=0.22.0`

### Projects 07 & 10: CrewAI-based (Conflicts with others)
- **Must use separate virtual environments**
- CrewAI requires:
  - `chromadb~=1.1.0` (conflicts with langchain-chroma)
  - `tokenizers~=0.20.3` (conflicts with transformers)
  - `beautifulsoup4~=4.13.4` (conflicts with newer versions)

### Project 08: OpenAI API (Minimal)
- Only needs `openai` package
- Can be installed with base requirements

### Project 09: Deep Q-Learning (ML-focused)
- Needs PyTorch, transformers
- Can conflict with CrewAI projects due to tokenizers

## Quick Reference

| Project | Key Dependencies | Conflicts With |
|---------|------------------|----------------|
| 01 | langchain-chroma, chromadb>=1.4.0 | CrewAI projects |
| 02 | langgraph | None |
| 03 | langchain-ollama, duckduckgo-search | None |
| 04 | duckduckgo-search | None |
| 05 | google-genai, langgraph | None |
| 06 | gymnasium, pygame | None |
| 07 | crewai, crewai-tools | Projects 01, 09 |
| 08 | openai | None |
| 09 | torch, transformers, tokenizers>=0.22.0 | CrewAI projects |
| 10 | crewai, crewai-tools, pillow, pydantic | Projects 01, 09 |

## Troubleshooting

### Conflict Resolution

If you encounter conflicts:

1. **Use separate virtual environments** (recommended)
2. **Install projects in groups**:
   - Group 1: Projects 01-06 (LangChain compatible)
   - Group 2: Projects 07, 10 (CrewAI - separate env)
   - Group 3: Project 09 (ML - separate env if using CrewAI)
   - Group 4: Project 08 (Minimal - can go anywhere)

### Common Conflicts

- **chromadb**: CrewAI needs 1.1.x, langchain-chroma needs >=1.3.5
- **tokenizers**: CrewAI needs 0.20.3, transformers needs >=0.22.0
- **beautifulsoup4**: CrewAI-tools needs 4.13.4, others may need >=4.14.0

## Updating Requirements

When adding new dependencies:

1. Add common dependencies to `base.txt`
2. Add project-specific dependencies to the appropriate `XX-project-name.txt`
3. Update this README if there are new conflicts

