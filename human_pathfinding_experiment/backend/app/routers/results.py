from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings
from app.schemas.experiment import ExperimentResult
from app.schemas.requests import ResultsRequest
from app.schemas.responses import ResultsResponse

router = APIRouter()


@router.post("/results", response_model=ResultsResponse)
async def save_results(request: ResultsRequest):
    """
    Save experiment result as parquet file.

    Args:
        request: Experiment result data.

    Returns:
        ResultsResponse: Saved filename and success message.
    """
    try:
        # Parse timestamp_start
        timestamp_start = datetime.fromisoformat(request.timestamp_start)

        # Create experiment result
        result = ExperimentResult(
            timestamp=request.timestamp,
            obstacle_mode=request.obstacle_mode,
            window_size=request.window_size,
            strategy=request.strategy,
            prediction_mode=request.prediction_mode,
            all_paths=request.all_paths,
            path=request.path,
            path_valid=request.path_valid,
            experiment_duration_seconds=request.experiment_duration_seconds,
            timestamp_start=timestamp_start,
            success=request.success,
            total_waypoints=request.total_waypoints,
            total_distance_m=request.total_distance_m,
        )

        # Generate filename
        filename = f"human_{request.timestamp}_{request.obstacle_mode}_w{request.window_size}.parquet"
        output_path = settings.results_path / filename

        # Ensure results directory exists
        settings.results_path.mkdir(parents=True, exist_ok=True)

        # Save to parquet
        df = result.to_dataframe()
        df.to_parquet(output_path, index=False)

        return ResultsResponse(
            filename=filename,
            message=f"Result saved successfully to {filename}",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving results: {str(e)}",
        )


@router.get("/results/{filename}")
async def download_results(filename: str):
    """
    Download a parquet result file.

    Args:
        filename: Name of the parquet file to download.

    Returns:
        FileResponse: The parquet file.
    """
    file_path = settings.results_path / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Result file not found: {filename}",
        )

    if not file_path.suffix == ".parquet":
        raise HTTPException(
            status_code=400,
            detail="Only parquet files can be downloaded",
        )

    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=filename,
    )


@router.get("/results")
async def list_results():
    """
    List all saved result files.

    Returns:
        dict: List of result filenames.
    """
    if not settings.results_path.exists():
        return {"results": []}

    results = [
        f.name
        for f in settings.results_path.glob("*.parquet")
        if f.is_file()
    ]

    return {"results": sorted(results)}
