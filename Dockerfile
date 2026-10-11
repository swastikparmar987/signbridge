FROM python:3.11-slim

# Install system dependencies for OpenCV (libgl1), FFmpeg, audio (libsndfile1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app

WORKDIR /app

# Install Python packages
COPY --chown=user requirements.txt /app/
RUN pip install --no-cache-dir --user -r requirements.txt \
    && pip install --no-cache-dir --user onnxruntime kokoro-onnx soundfile httpx

# Copy repository code
COPY --chown=user . /app

EXPOSE 7860

CMD ["uvicorn", "signbridge.web.app:app", "--host", "0.0.0.0", "--port", "7860"]
