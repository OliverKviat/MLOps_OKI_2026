# Base image with Python 3.12 and uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install system dependencies for building Python packages
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy project configuration and source files
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/

# Set working directory
WORKDIR /

# Create directories for models and data mounts
RUN mkdir -p models

# Configure uv to use copy mode for faster builds
ENV UV_LINK_MODE=copy

# Install dependencies with cache mount for faster rebuilds
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked

# Set the entry point to run the evaluation script
ENTRYPOINT ["uv", "run", "src/adv_cookie_recipy/evaluate.py"]
