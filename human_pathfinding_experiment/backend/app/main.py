from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.config import settings
from app.routers import ab_points, admin, obstacles, participants, results, timestamps, trials


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Ensure results directory exists
    settings.results_path.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Human Pathfinding Experiment API",
    description="Backend API for human pathfinding experiment web application",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS for the frontend (explicit origin allowlist, see CORS_ORIGINS env)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compress large GeoJSON responses (an obstacles payload shrinks ~7x) for slow links.
# Low compresslevel: JSON compresses well anyway, and CPU is scarce on small plans
# (level 9 measured ~1-2 s per request on Render free)
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=1)

# Include routers
app.include_router(timestamps.router, prefix="/api", tags=["timestamps"])
app.include_router(ab_points.router, prefix="/api", tags=["ab-points"])
app.include_router(obstacles.router, prefix="/api", tags=["obstacles"])
app.include_router(results.router, prefix="/api", tags=["results"])
app.include_router(participants.router, prefix="/api", tags=["participants"])
app.include_router(trials.router, prefix="/api", tags=["trials"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
