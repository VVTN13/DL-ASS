# One image, two commands: the API (`fastapi run`) and the worker (`python -m app.worker`).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# Dependencies change rarely: install them before copying the source for a cached layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

COPY alembic.ini ./
COPY alembic ./alembic
COPY samples ./samples
COPY app ./app

# Checkpoints are mounted read-only at /checkpoints; uploads live in the /data volume.
RUN useradd --create-home app && mkdir /data && chown app /data
USER app
ENV DATA_DIR=/data CHECKPOINTS_DIR=/checkpoints
EXPOSE 8000
CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
