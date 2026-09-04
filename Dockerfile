FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
ARG MODEL_EXTRAS=""
RUN uv sync --frozen --no-dev --no-cache ${MODEL_EXTRAS}
COPY configs ./configs
COPY scripts ./scripts
RUN useradd --create-home app && mkdir -p data models artifacts && chown -R app:app data models artifacts
USER app
ENV PATH="/app/.venv/bin:$PATH"
ENV FREIGHTLENS_CONFIG="configs/base.yaml"
EXPOSE 8000
CMD ["uvicorn", "freightlens.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
