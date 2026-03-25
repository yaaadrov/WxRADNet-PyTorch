from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import ab_points, obstacles, results, timestamps


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

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(timestamps.router, prefix="/api", tags=["timestamps"])
app.include_router(ab_points.router, prefix="/api", tags=["ab-points"])
app.include_router(obstacles.router, prefix="/api", tags=["obstacles"])
app.include_router(results.router, prefix="/api", tags=["results"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
