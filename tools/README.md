# Tools

This directory contains utility scripts for the AI Agent Projects.

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

## Notes

- Both scripts automatically detect the project root directory
- They work regardless of where you run them from
- The scripts use relative paths from the project root

