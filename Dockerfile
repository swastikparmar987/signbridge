FROM node:20-slim AS web_builder
WORKDIR /build/signbridge-web
COPY signbridge-web/package.json signbridge-web/package-lock.json* ./
RUN npm ci
COPY signbridge-web/ ./
RUN npm run build

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=web_builder /usr/local/bin/node /usr/local/bin/node
COPY --from=web_builder /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# Non-root user compatible with Hugging Face Spaces / Cloud containers
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app \
    SIGNBRIDGE_HOST=0.0.0.0

WORKDIR /app

COPY --chown=user requirements-docker.txt /app/
RUN pip install --no-cache-dir --user --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0" \
    && pip install --no-cache-dir --user -r requirements-docker.txt

COPY --chown=user . /app
COPY --from=web_builder --chown=user /build/signbridge-web /app/signbridge-web

RUN mkdir -p /app/trained_models/tts /app/data/personalization /app/signbridge/web/static/audio \
    && if [ ! -f /app/trained_models/tts/kokoro-v0_19.onnx ]; then \
         curl -fsSL --retry 3 -o /app/trained_models/tts/kokoro-v0_19.onnx \
           https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx; \
       fi \
    && if [ ! -f /app/trained_models/tts/voices.bin ]; then \
         curl -fsSL --retry 3 -o /app/trained_models/tts/voices.bin \
           https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin; \
       fi \
    && chmod +x /app/scripts/start_app.sh

ENV PORT=3099
EXPOSE 3099 8001 7860 8080

CMD ["/app/scripts/start_app.sh"]
