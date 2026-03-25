from fastapi import APIRouter, HTTPException, Query

from app.schemas.responses import ABPointsResponse
from app.services.data_service import DataService

router = APIRouter()

# Shared service instance
_data_service = DataService()


@router.get("/ab-points", response_model=ABPointsResponse)
async def get_ab_points(
    timestamp: str = Query(..., description="Timestamp directory name"),
):
    """
    Get A and B points for the selected timestamp.

    Args:
        timestamp: Timestamp directory name.

    Returns:
        ABPointsResponse: A and B point coordinates.
    """
    try:
        a_point, b_point = _data_service.get_ab_points_for_timestamp(timestamp)
        return ABPointsResponse(
            a_point=list(a_point),
            b_point=list(b_point),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading AB points: {str(e)}",
        )
