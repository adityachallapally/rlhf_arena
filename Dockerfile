# RLHF Arena Docker Image
# Multi-stage build for optimized production image

# Base stage with CUDA support
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN ln -s /usr/bin/python3.9 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies for better performance
RUN pip install --no-cache-dir \
    flash-attn \
    xformers \
    bitsandbytes \
    accelerate

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/reports /workspace/experiments /workspace/logs /workspace/checkpoints

# Set permissions
RUN chmod +x scripts/*.py

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

# Default command
CMD ["python", "scripts/benchmark.py", "--help"]

# Labels
LABEL maintainer="RLHF Arena Team"
LABEL version="0.1.0"
LABEL description="RLHF Arena: Benchmarking frontier post-training RL methods for LLMs"
LABEL org.opencontainers.image.source="https://github.com/your-org/rlhf_arena"
