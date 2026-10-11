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
COPY --chown=user requirements-docker.txt /app/
RUN pip install --no-cache-dir --user -r requirements-docker.txt \
    && pip install --no-cache-dir --user onnxruntime kokoro-onnx soundfile httpx

# Copy repository code
COPY --chown=user . /app

ENV PORT=8000
EXPOSE 8000 7860 8080

CMD ["sh", "-c", "uvicorn signbridge.web.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
