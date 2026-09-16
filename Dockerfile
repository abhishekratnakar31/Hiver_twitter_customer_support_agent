FROM python:3.11-slim

WORKDIR /app

# Configure environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    LOKY_MAX_CPU_COUNT=1 \
    JOBLIB_MULTIPROCESSING=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# Install system utilities (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, models, configs, and web assets
COPY configs/ ./configs/
COPY models/ ./models/
COPY src/ ./src/
COPY web/ ./web/
COPY data/ ./data/
COPY reports/ ./reports/

# Expose application port
EXPOSE 8000

# Define container healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch Uvicorn FastAPI server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
