FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements-full.txt .
RUN pip install --no-cache-dir -r requirements-full.txt

# Copy source code
COPY . .

# Expose ports: FastAPI=8000, Streamlit=8501
EXPOSE 8000 8501

# Health check for FastAPI
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI backend in background and Streamlit on cloud PORT
CMD ["sh", "-c", "uvicorn api.main:app --host 127.0.0.1 --port 8000 & streamlit run ui/dashboard.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
