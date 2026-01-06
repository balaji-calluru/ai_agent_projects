# Tools

This directory contains utility scripts and helper tools for the AI Agent Projects.

**Note:** All helper scripts (e.g., `create_*.py`, `download_*.py`, `generate_*.py`) should be saved in this `tools/` folder for better organization and maintainability.

## Scripts

### `create_agent_gif.py`

Creates an animated GIF visualization of a multi-agent system showing orchestration and data flow.

**Usage:**
```bash
python3 tools/create_agent_gif.py
# or
cd tools && python3 create_agent_gif.py
```

**Output:**
- Creates `data/Agent_Output/images/agent-orchestration.gif`
- Shows 5 agents arranged around an orchestrator
- Animates data flow and agent activity
- 3-second animation with 30 frames

**Requirements:**
- `PIL` (Pillow)
- `imageio`

### `launch_jupyter.sh`

Launches Jupyter Notebook server for viewing and editing notebooks.

**Usage:**
```bash
./tools/launch_jupyter.sh
# or
cd tools && ./launch_jupyter.sh
```

**Features:**
- Automatically activates the virtual environment
- Changes to project root directory
- Opens Jupyter Notebook in your browser
- Works from any directory (uses script location to find project root)

**Requirements:**
- Virtual environment at `venv/` in project root
- Jupyter Notebook installed

## Helper Scripts Convention

**All helper scripts should be saved in the `tools/` folder:**
- Image generation scripts: `create_*.py`, `generate_*.py`
- Data download scripts: `download_*.py`, `fetch_*.py`
- Utility scripts: `*_helper.py`, `*_util.py`
- Any temporary or one-off scripts

**Examples:**
- ✅ `tools/create_images.py` - Creates diagrams for documentation
- ✅ `tools/download_data.py` - Downloads datasets
- ✅ `tools/create_agent_gif.py` - Generates animated GIFs
- ❌ `src/create_images.py` - Should be in tools/
- ❌ `create_images.py` - Should be in tools/

**Benefits:**
- Better organization and discoverability
- Keeps `src/` folder clean (only notebooks and main code)
- Easier to share and maintain utility scripts
- Clear separation between project code and helper tools

## Notes

- Both scripts automatically detect the project root directory
- They work regardless of where you run them from
- The scripts use relative paths from the project root
- Temporary outputs from helper scripts are ignored by git (see `.gitignore`)

