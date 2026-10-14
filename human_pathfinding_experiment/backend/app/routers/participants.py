from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from app.db.engine import get_session
from app.db.models import Participant
from app.routers.auth import get_current_participant
from app.schemas.requests import LoginRequest, QuestionnaireRequest
from app.schemas.responses import LoginResponse
from app.services.auth_service import AuthService
from app.services.participant_service import ParticipantService, ParticipantError


router = APIRouter(prefix="/participants", tags=["participants"])


@router.post("/demo")
def create_demo_session() -> dict[str, str]:
    """Issue an anonymous demo token: nothing is stored in the database."""
    return {"token": AuthService.create_demo_token()}


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest, session: Session = Depends(get_session)) -> LoginResponse:
    """Log the participant in; create the participant on the fly if it is the first visit."""
    service = ParticipantService(session)
    try:
        participant = service.login(surname=request.surname.strip(), name=request.name.strip(), group=request.group.strip())
    except ParticipantError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message) from None

    return LoginResponse(
        token=AuthService.create_token(participant.id),  # type: ignore[arg-type]
        participant_id=participant.id,  # type: ignore[arg-type]
        is_completed=participant.is_completed,
    )


@router.post("/me/questionnaire")
def submit_questionnaire(
    request: QuestionnaireRequest,
    participant: Participant = Depends(get_current_participant),
    session: Session = Depends(get_session),
) -> dict[str, bool]:
    """Store the final questionnaire answers after all trials are submitted."""
    service = ParticipantService(session)
    try:
        service.save_questionnaire(participant, request)
    except ParticipantError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message) from None
    return {"is_completed": True}
