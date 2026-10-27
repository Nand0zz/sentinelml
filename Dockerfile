FROM python:3.12-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
# Runtime + DVC (S3 remote) only; no test/dev dependencies.
RUN pip install --no-cache-dir ".[dvc]"
COPY scripts/start_api.sh ./scripts/start_api.sh

# The model is NOT baked into the image. At start, scripts/start_api.sh either serves a model
# mounted at $MODEL_DIR, or (DVC_REPO_URL set) fetches the latest champion from the DVC remote:
#   docker run -p 8000:8000 -v "$PWD/models/champion:/models/champion:ro" sentinelml-api
#   docker run -p 8000:8000 -e DVC_REPO_URL=https://github.com/<user>/sentinelml -e AWS_... \
#              -v sentinelml-model:/models/champion sentinelml-api
ENV MODEL_DIR=/models/champion \
    PREDICTION_LOG=/app/logs/predictions.jsonl \
    PYTHONUNBUFFERED=1

RUN chmod +x scripts/start_api.sh \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /app/logs /models/champion && chown -R app /app/logs /models
USER app
EXPOSE 8000

# start-period covers the model download on first boot
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

ENTRYPOINT ["./scripts/start_api.sh"]
