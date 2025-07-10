# Multi-stage build for smaller image size
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements and create optimized requirements for production
COPY ./requirements.txt /build/requirements.txt

# Create production requirements without dev dependencies
RUN grep -v -E "(pre-commit|ruff)" /build/requirements.txt > /build/requirements-prod.txt

# Install dependencies with CPU-only PyTorch
RUN pip install --no-cache-dir --upgrade \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision && \
    pip install --no-cache-dir --upgrade -r /build/requirements-prod.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY ./src /code/src
COPY ./main.py /code/

EXPOSE 8002

CMD ["python", "-m", "main", "--port", "8002"]
