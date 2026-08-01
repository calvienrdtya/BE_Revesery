# syntax=docker/dockerfile:1
FROM python:3.12-slim

# scikit-learn/shap butuh libgomp untuk OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root + data dir sebagai volume (persist result.json + latest.xlsx).
RUN useradd -m app && mkdir -p /app/data && chown -R app /app
USER app
VOLUME ["/app/data"]

ENV PORT=8000 DATA_DIR=/app/data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\",\"8000\")}/health')" || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
