# Use CUDA-enabled Python base image for GPU support
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV SAM2_BUILD_CUDA=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Clone the NukeSamurai repository if it's a submodule
# This is needed because Docker doesn't automatically include git submodules
RUN mkdir -p NukeSamurai
RUN if [ ! -d "NukeSamurai/sam2_repo" ]; then \
    git clone https://github.com/facebookresearch/segment-anything-2.git NukeSamurai/sam2_repo; \
    fi

# Copy requirements and setup files first for caching
COPY requirements.txt setup.py ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . /app

# Install the application in development mode
RUN pip install -e .

# Install SAM2 repo in editable mode
RUN cd NukeSamurai/sam2_repo && \
    pip install -e . && \
    pip install -e ".[notebooks]"

# Download SAM2 model checkpoints
RUN mkdir -p NukeSamurai/sam2_repo/sam2/checkpoints && \
    cd NukeSamurai/sam2_repo/sam2/checkpoints && \
    if [ -f "download_ckpts.sh" ]; then \
        chmod +x download_checkpoints.sh && \
        ./download_checkpoints.sh; \
    else \
        echo "Downloading checkpoints directly..."; \
        wget -q https://dl.fbaipublicfiles.com/segment_anything_2/sam2_tiny.pt || echo "Failed to download tiny model"; \
        wget -q https://dl.fbaipublicfiles.com/segment_anything_2/sam2_b.pt || echo "Failed to download base model"; \
    fi

# Create output directory for API results
RUN mkdir -p Output

# Expose FastAPI port
EXPOSE 8000

# Create a healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command to run the API server using our CLI
ENTRYPOINT ["nuke-samurai-server"]
CMD ["--host", "0.0.0.0", "--port", "8000"] 