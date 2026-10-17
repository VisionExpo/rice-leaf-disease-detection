FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app

WORKDIR /app

# Runtime library required by TensorFlow.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install locked production dependencies.
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

# Copy application code only.
COPY api ./api
COPY src ./src

# Run as a non-root user.
RUN useradd --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/models \
    && chown -R app:app /app


USER app

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]