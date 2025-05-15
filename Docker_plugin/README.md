# NukeSamurai Backend

Backend server for the NukeSamurai plugin, providing SAM2 (Segment Anything Model 2) functionality through a REST API.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Git
- pip 20.0 or higher (for dependency management)

## Installation

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/your-repo/NPP.git
cd NPP/Docker_plugin
```

2. Create and activate a virtual environment:
```bash
# On Linux/Mac:
python -m venv .venv
source .venv/bin/activate

# On Windows:
python -m venv .venv
.venv\Scripts\activate
```

3. Upgrade pip and install build tools:
```bash
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel
```

4. Install the package in development mode:
```bash
pip install -e .
```

Note: Do NOT run `python setup.py` directly. Always use `pip install -e .` as it:
- Properly handles dependency resolution
- Installs required build tools
- Sets up the package in development mode
- Runs all necessary installation steps in the correct order

The installation will:
- Check Python version compatibility
- Install all required dependencies
- Set up the SAM2 repository
- Create default configuration files
- Set up command-line tools

5. Verify the installation:
```bash
# Check if the command-line tools are installed
nuke-samurai-server --version

# Check if SAM2 is properly installed
python -c "from sam2.utils.transforms import ResizeLongestSide; print('SAM2 installed successfully')"
```

## Configuration

The default configuration is created in `configs/config.yaml`. You can modify it to change:
- API host and port
- Log level
- Output directory
- CUDA device
- Batch size

Example configuration:
```yaml
api_host: "0.0.0.0"
api_port: 8000
log_level: "INFO"
output_dir: "Output"
max_batch_size: 32
cuda_device: 0  # Set to -1 for CPU
```

## Running the Server

1. Start the API server:
```bash
nuke-samurai-server
```

2. The server will be available at `http://localhost:8000`

3. Check the API is running:
```bash
curl http://localhost:8000/api/v1/health
```

## Development

1. Run tests:
```bash
nuke-samurai-test
```

2. Format code:
```bash
black src/
isort src/
```

## Troubleshooting

1. If you get CUDA errors:
   - Check your CUDA installation
   - Try setting `cuda_device: -1` in config.yaml to use CPU
   - Verify PyTorch is installed with CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

2. If SAM2 installation fails:
   - Check the SAM2 submodule is properly cloned:
     ```bash
     git submodule update --init --recursive
     ```
   - Try installing manually:
     ```bash
     cd NukeSamurai/sam2_repo
     pip install -e .
     pip install -e .[notebooks]
     ```
   - Check for compiler errors in the logs

3. If the server won't start:
   - Check the port is not in use: `netstat -ano | findstr :8000` (Windows) or `lsof -i :8000` (Linux)
   - Check the logs in `logs/` directory
   - Verify all dependencies are installed: `pip freeze`
   - Try running with debug logging: `nuke-samurai-server --log-level debug`

4. Common Installation Issues:
   - If you get "command not found" errors, make sure your virtual environment is activated
   - If you get build errors, install required system packages:
     ```bash
     # Ubuntu/Debian:
     sudo apt-get update
     sudo apt-get install python3-dev build-essential

     # CentOS/RHEL:
     sudo yum groupinstall "Development Tools"
     sudo yum install python3-devel
     ```

## API Documentation

See [API.md](../Wikis/API.md) for detailed API documentation.

## License

MIT License - see LICENSE file for details 