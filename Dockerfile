# Multi-stage Dockerfile for Deep Learning Model Scaling Analysis

# Builder stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml setup.py README.md LICENSE ./
COPY src/ ./src/

# Build wheel
RUN pip install --no-cache-dir build && \
    python -m build --wheel

# Runtime stage
FROM python:3.11-slim

LABEL maintainer="Deep Learning Model Scaling Analysis Contributors"
LABEL description="Causal inference analysis of neural network scaling laws"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheel from builder and install
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Create data directory
RUN mkdir -p /app/data /app/outputs

# Set environment variables
ENV DML_DATA_DIR=/app/data
ENV DML_OUTPUT_DIR=/app/outputs
ENV PYTHONUNBUFFERED=1

# Default command
ENTRYPOINT ["dml-scale"]
CMD ["--help"]
