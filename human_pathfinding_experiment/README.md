# Human Pathfinding Experiment Web Application

This application allows users to manually navigate aircraft through time-varying thunderstorm obstacles, recording their pathfinding decisions for analysis.

## Prerequisites

- Docker and Docker Compose
- Data files in `../data/` directory
- Configuration files in `../config/` directory:
  - `timestamps.pkl` - List of timestamp objects
  - `ab_points.pkl` - List of (A_point, B_point) tuples

## Quick Start

```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

## Access

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/timestamps` | List available timestamp directories |
| GET | `/api/ab-points?timestamp=...` | Get A and B points for timestamp |
| POST | `/api/obstacles` | Get obstacles for current position |
| POST | `/api/results` | Save experiment result |
| GET | `/api/results` | List saved results |
| GET | `/api/results/{filename}` | Download result file |

## User Flow

1. **Select Timestamp**: Choose from available timestamp directories
2. **Select Obstacle Mode**:
   - `obstacles` - Raw obstacle polygons only
   - `hull` - Raw obstacles + concave/convex hull
3. **Select Window Size**:
   - `1` - Current time only
   - `7` - Current time + 6 predictions
4. **Navigate**: Click on canvas to place waypoints
   - Click near B point to complete path
   - Arrow keys add/remove prediction layers

## Keyboard Controls

| Key | Action |
|-----|--------|
| Arrow Right | Add next prediction layer (window_size=7) |
| Arrow Left | Remove last prediction layer |
| 1 | Set window size to 1 |
| 7 | Set window size to 7 |

## Results

Results are saved as Parquet files containing:
- Timestamp and configuration
- All waypoints and segmented path
- Path validity
- Experiment duration
- Total distance traveled

## Development

### Backend

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Architecture

```
human_pathfinding_experiment/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI entry point
│   │   ├── config.py        # Pydantic settings
│   │   ├── routers/         # API endpoints
│   │   ├── services/        # Business logic (reuses thund_avoider)
│   │   └── schemas/         # Pydantic models
│   ├── pyproject.toml
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── hooks/           # Custom hooks
│   │   ├── services/        # API client
│   │   ├── types/           # TypeScript types
│   │   └── utils/           # Utilities
│   ├── package.json
│   └── Dockerfile
└── docker-compose.yml
```

## Code Reuse

This application reuses logic from `thund_avoider` modules:

- `MaskedPreprocessor` - Oriented bounding boxes, polygon clipping
- `DataLoader` - Time key extraction, obstacle loading
- `is_line_valid()` - Path validation
- Configuration constants - Velocity, delta, distances
