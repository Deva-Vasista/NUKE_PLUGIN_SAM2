# Use official Python base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy backend code
COPY . /app

# Install SAM2 repo in editable mode
RUN pip install -e NukeSamurai/sam2_repo

# Expose FastAPI port
EXPOSE 8000

# Default command to run the API server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"] 