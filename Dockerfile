# API container: FastAPI + retrieval (bge-m3, reranker on CPU) + the vLLM client.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    EMBED_DEVICE=cpu \
    RERANK_DEVICE=cpu

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# torch as a CPU wheel (no CUDA pulled into the API container)
RUN pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cpu

COPY requirements.api.txt .
RUN pip install -r requirements.api.txt

COPY src ./src

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
  CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
