# Hugging Face Spaces Docker template: 2 vCPU / 16 GB RAM on the free tier.
# Built by scripts/prepare_hf_space.py (build context = deploy/hf_space).
FROM python:3.12-slim

# Geospatial libs for geopandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires a non-root user
RUN useradd -m -u 1000 user

WORKDIR /app

COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY pyproject.toml .
COPY scripts/ ./scripts/

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
    pyjwt>=2.15.1

# 16 GB RAM: warm all 11 timestamps at startup (~1.3 GB) so no request ever
# hits the one-time CSV parse
ENV WARM_CACHE=1
ENV OBSTACLE_CACHE_SIZE=16

RUN chown -R user:user /app
USER user

# Download the dataset parts (DATA_ARCHIVE_URL secret), migrate, seed, serve on 7860
CMD ["sh", "-c", "python scripts/fetch_data.py && alembic upgrade head && python scripts/seed_experiment_data.py && uvicorn app.main:app --host 0.0.0.0 --port 7860"]
