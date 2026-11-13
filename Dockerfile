FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REGENREC_HOST=0.0.0.0 \
    REGENREC_PORT=8000

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN python -m pip install --upgrade pip && \
    python -m pip install -e '.[serving]'

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"

CMD ["python", "-m", "uvicorn", "regenrec.api:app", "--host", "0.0.0.0", "--port", "8000"]
