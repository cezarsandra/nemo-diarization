# CUDA 11.8 runtime — supports Pascal (GTX 1070, sm_61) through Hopper (H100, sm_90)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
        python3.10 python3-pip \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip

WORKDIR /app

# 1 — Install all Python deps (nemo_toolkit will pull its own cpu torch here)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2 — Reinstall PyTorch cu118 AFTER nemo so it doesn't get overwritten
RUN pip install --no-cache-dir \
        torch torchaudio \
        --index-url https://download.pytorch.org/whl/cu118

# 3 — Copy application
COPY diarize.py handler.py config.yaml ./

# 4 — Pre-download NeMo models so RunPod cold starts don't hit NGC at runtime
RUN python3 -c "\
import nemo.collections.asr as nemo_asr; \
nemo_asr.models.EncDecClassificationModel.from_pretrained('vad_multilingual_marblenet'); \
nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained('titanet_large'); \
nemo_asr.models.EncDecDiarLabelModel.from_pretrained('diar_msdd_telephonic'); \
print('Models pre-downloaded OK')"

CMD ["python3", "handler.py"]
