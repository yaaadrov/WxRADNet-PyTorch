from fastapi import APIRouter, HTTPException

from app.schemas.responses import TimestampsResponse
from app.services.data_service import DataService

router = APIRouter()

# Shared service instance
_data_service = DataService()


@router.get("/timestamps", response_model=TimestampsResponse)
async def get_timestamps():
    """
    Get list of available timestamp directories from data path.

    Returns:
        TimestampsResponse: List of available timestamp directory names.
    """
    try:
        timestamps = _data_service.get_available_timestamps()
        return TimestampsResponse(timestamps=timestamps)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading timestamps: {str(e)}",
        )
