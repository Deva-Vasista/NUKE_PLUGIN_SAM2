# NukeSAM2 Plugin

A Nuke plugin for rotoscopy using Meta's SAM2 API.

## Dependencies

The plugin requires the following Python packages:
- opencv-python
- requests
- websockets

## Setup Instructions

1. Make sure you have Python 3.7+ installed on your system.

2. Run the setup script to create a virtual environment and install dependencies:
   ```bash
   python setup.py
   ```

3. Copy the entire `Nuke_plugin` directory to your Nuke plugins directory:
   - Windows: `C:\Users\<username>\AppData\Roaming\Nuke\<version>\plugins`
   - Linux: `~/.nuke/plugins`
   - Mac: `~/Library/Application Support/Nuke/<version>/plugins`

4. Restart Nuke.

5. The SAM2 node will be available in the Nodes menu.

## Usage

1. Create a SAM2 node in your Nuke script
2. Connect it to your input sequence
3. Click "Update Path" to set the input file path
4. Click "Create Bounding Box" to select the area to mask
5. Set your desired frame range and output settings
6. Click "Generate Mask" to start the process

## Troubleshooting

If you encounter any issues:

1. Make sure the virtual environment is properly set up by running `setup.py`
2. Check that the backend API server is running at `http://localhost:8000`
3. Verify that all dependencies are installed in the virtual environment
4. Check the Nuke console for any error messages

## Notes

- The plugin requires a running SAM2 API server
- Input must be an image sequence (not video files)
- Output can be either EXR sequence or MP4
- Progress is shown in the node's UI during processing 