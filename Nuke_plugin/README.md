# NukeSamurai Nuke Plugin

A Nuke plugin for advanced rotoscoping and object segmentation using Meta's SAM2 model, powered by the NukeSamurai backend API.

---

## Introduction
This plugin brings state-of-the-art, AI-powered segmentation and tracking to Nuke. It connects to the NukeSamurai backend (FastAPI server) and allows you to:
- Generate masks for EXR image sequences
- Use bounding boxes and point prompts
- Track objects across frames
- Export masks as EXR or MP4

---

## Installation

### 1. Copy the Plugin
Copy the entire `Nuke_plugin` folder into your Nuke plugins directory:
- **Windows:** `C:\Users\<username>\AppData\Roaming\Nuke\<vers ion>\plugins\Nuke_plugin`
- **Linux:** `~/.nuke/plugins/Nuke_plugin`
- **Mac:** `~/Library/Application Support/Nuke/<version>/plugins/Nuke_plugin`

Your structure should look like:
```
Nuke
└── plugins/
    └── Nuke_plugin/
        ├── scripts/
        ├── icons/
        ├── ...
```

### 2. Add to Nuke's Plugin Path
In your Nuke `plugins/init.py` (not in `Nuke_plugin/init.py`), add:
```python
nuke.pluginAddPath('Nuke_plugin')
```
This ensures Nuke loads the plugin at startup.

### 3. Set Up Python Environment
It is **strongly recommended** to use a virtual environment for dependencies:
```bash
cd Nuke_plugin
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```
- The `init.py` in `Nuke_plugin` will automatically add the venv's `site-packages` to `sys.path`.
- Alternatively, you can run:
  ```bash
  python setup.py develop
  # or
  pip install -e .
  ```

### 4. Start the Backend
Make sure the NukeSamurai backend API server is running (see backend README).

---

## Usage
1. Launch Nuke. The SAM2 node will appear in the Nodes menu.
2. Create a SAM2 node in your script.
3. Connect it to your input image sequence.
4. Click **Update Path** to set the input file path.
5. Click **Create Selection** to draw bounding boxes or add points.
6. Set frame range, model type, and output settings.
7. Click **Generate Mask** to start processing.
8. Progress and status will be shown in the node UI.

---

## Troubleshooting
- **Virtual environment not detected:** Ensure you activated the venv before launching Nuke, or that `init.py` correctly adds the venv's `site-packages`.
- **Backend not running:** Start the backend API server (`nuke-samurai-server`) and ensure it's accessible at `http://localhost:8000`.
- **Missing dependencies:** Run `pip install -r requirements.txt` in your venv.
- **No node in menu:** Double-check `nuke.pluginAddPath('Nuke_plugin')` is in your `plugins/init.py`.
- **Input errors:** Only image sequences (not video files) are supported.
- **Output issues:** Ensure output directory is writable.
- **See Nuke console:** For detailed error messages.

---

## Notes
- Requires a running NukeSamurai backend API server
- Supports EXR input/output and MP4 output
- Compatible with Nuke 13+ (tested on Linux, Windows, Mac)
- For more, see the [official NukeSamurai repo](https://github.com/Theo-SAMINADIN-td/NukeSamurai) 