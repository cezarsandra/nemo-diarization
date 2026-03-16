# CUDA 11.8 runtime — supports Pascal (GTX 1070, sm_61) through Hopper (H100, sm_90)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
        python3.10 python3-pip \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch with CUDA 11.8 (must come before nemo_toolkit to avoid cpu-only torch)
RUN pip install --no-cache-dir \
        torch torchaudio \
        --index-url https://download.pytorch.org/whl/cu118

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY diarize.py handler.py config.yaml ./

CMD ["python3", "handler.py"]
