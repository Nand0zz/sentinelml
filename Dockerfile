FROM python:3.12-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# The model is NOT baked into the image: mount an exported folder at /models/champion
#   docker run -p 8000:8000 -v "$PWD/exported_model:/models/champion:ro" sentinelml-api
ENV MODEL_DIR=/models/champion \
    PREDICTION_LOG=/app/logs/predictions.jsonl \
    PYTHONUNBUFFERED=1

RUN useradd --create-home app && mkdir -p /app/logs && chown app /app/logs
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "sentinelml.api:app", "--host", "0.0.0.0", "--port", "8000"]
