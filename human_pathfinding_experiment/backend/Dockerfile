FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml first
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir .

# Copy thund_avoider module (will be mounted at runtime, but we need dependencies)
RUN pip install --no-cache-dir \
    geopandas>=1.0.0 \
    shapely>=2.0.0 \
    pyarrow>=18.0.0 \
    numpy>=1.26.0 \
    pandas>=2.0.0 \
    networkx>=3.0 \
    pydantic>=2.0.0 \
    pydantic-settings>=2.0.0

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]