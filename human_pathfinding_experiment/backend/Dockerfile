FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code, migrations and seed data
# (build context is the human_pathfinding_experiment directory, see docker-compose.yml)
COPY backend/app/ ./app/
COPY backend/alembic/ ./alembic/
COPY backend/alembic.ini .
COPY backend/pyproject.toml .
COPY backend/scripts/ ./scripts/
COPY timestamp_selection/ranking.csv ./seed/ranking.csv

# Install Python dependencies
RUN pip install --no-cache-dir . && pip install --no-cache-dir \
    geopandas>=1.0.0 \
    shapely>=2.0.0 \
    pyarrow>=18.0.0 \
    numpy>=1.26.0 \
    pandas>=2.0.0 \
    networkx>=3.0 \
    pydantic>=2.0.0 \
    pydantic-settings>=2.0.0 \
    sqlmodel>=0.0.48 \
    alembic>=1.20.0 \
    psycopg2-binary>=2.9.13 \
    pyjwt>=2.9.0

# Fetch the dataset archive (skipped when already present/mounted), apply migrations and
# seed reference data, then start the API
CMD ["sh", "-c", "python scripts/fetch_data.py && alembic upgrade head && python scripts/seed_experiment_data.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
