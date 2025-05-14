# NukeSamurai API

A FastAPI backend for processing EXR images and sequences with SAM2 segmentation models.

## Features

- Process single EXR files with bounding box and point prompts
- Process EXR sequences with multi-object, multi-frame tracking
- Support for reverse tracking
- GPU acceleration with fallback to CPU
- Docker support for easy deployment

## Setup Options

### Option 1: Docker (Recommended)

The easiest way to get started is with Docker:

```bash
# Clone the repository (if you haven't already)
git clone --recursive https://your-repository-url.git
cd Docker_plugin

# Build and run with Docker Compose (GPU mode)
docker-compose up -d

# For CPU-only mode (no GPU)
docker-compose --profile cpu up -d nuke-samurai-api-cpu
```

Note: The `--recursive` flag ensures that the NukeSamurai submodule is also cloned. If you've already cloned without this flag, run:

```bash
git submodule update --init --recursive
```

### Option 2: Local Installation

For local development:

```bash
# Clone the repository with submodules
git clone --recursive https://your-repository-url.git
cd Docker_plugin

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package and dependencies
pip install -e .

# Download SAM2 model checkpoints
cd NukeSamurai/sam2_repo/sam2/checkpoints
chmod +x download_checkpoints.sh
./download_checkpoints.sh
cd ../../../../

# Run the server
nuke-samurai-server --reload
```

## Usage

Once the server is running, you can access the API at http://localhost:8000/api/v1/

### API Endpoints

See the [API Reference](../Wikis/backend_api_reference.md) for detailed documentation of all endpoints.

### Example Curl Commands

Process a single EXR file with a bounding box:

```bash
curl -X POST "http://localhost:8000/api/v1/process_exr?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/data/frame_0001.exr",
    "bbox": [454, 185, 500, 633],
    "bits": "32-bit float"
  }' \
  --output mask.exr
```

Process a sequence with reverse tracking:

```bash
curl -X POST "http://localhost:8000/api/v1/process_sequence?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence_path": "/data/frame_%04d.exr",
    "frame_range": [1, 20],
    "reverse": true,
    "prompts": [
      { "frame_index": 15, "bbox": [454, 185, 500, 633] }
    ],
    "bits": "32-bit float"
  }' \
  --output masks.zip
```

## Running Tests

```bash
# Run all tests
nuke-samurai-test --all

# Run only API tests
nuke-samurai-test --api-tests

# Run only unit tests
nuke-samurai-test --unit-tests
```

## Environment Variables

When using Docker, you can customize the deployment with these environment variables:

- `EXR_DATA_PATH`: Path to your EXR files (default: ./Test_files)
- `OUTPUT_PATH`: Path for output files (default: ./Output)
- `CHECKPOINTS_PATH`: Path for model checkpoints (default: ./checkpoints)
- `CUDA_VISIBLE_DEVICES`: GPU device ID to use (default: 0, set to empty for CPU)
- `DEBUG`: Enable debug logging (default: false)
- `SAM2_BUILD_CUDA`: Enable CUDA builds (default: 1, set to 0 for CPU-only)

Example:

```bash
EXR_DATA_PATH=/path/to/exr/files OUTPUT_PATH=/path/to/output docker-compose up -d
```

## License

[MIT License](LICENSE) 