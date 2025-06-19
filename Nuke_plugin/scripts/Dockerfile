# ---- Builder stage (for pip cache, optional) ----
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS builder

WORKDIR /app

COPY requirements.txt setup.py ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- Final stage ----
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV SAM2_BUILD_CUDA=1
ENV PYTHONPATH=/app:/app/NukeSamurai/sam2_repo
ENV HYDRA_FULL_ERROR=1
ENV OPENCV_IO_ENABLE_OPENEXR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential libgl1 libglib2.0-0 ffmpeg git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Copy code and checkpoints (everything you need)
COPY . /app

# Install Python deps (from builder cache if possible)
COPY --from=builder /root/.cache /root/.cache

RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install -e .
RUN pip install -e NukeSamurai/sam2_repo
RUN pip install -e NukeSamurai/sam2_repo[notebooks]

# Make sure checkpoints are present
RUN test -f NukeSamurai/sam2_repo/checkpoints/sam2.1_hiera_base_plus.pt

RUN mkdir -p Output

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["nuke-samurai-server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
