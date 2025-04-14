# Use the official Python image
FROM python:3.10

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir numpy==2.0.0
RUN pip install --no-cache-dir --no-deps -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/raw \
    /app/data/processed \
    /app/model \
    /app/monitoring \
    /app/monitoring/visualizations \
    /app/mlruns \
    /app/src/frontend

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["bash", "-c", "streamlit run src/frontend/app.py & python src/pipeline.py && wait"]
