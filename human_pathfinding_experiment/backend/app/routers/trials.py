from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.config import settings
from app.db.engine import get_session
from app.db.models import Participant, Trial
from app.routers.auth import get_current_participant
from app.schemas.requests import SubmitTrialRequest
from app.schemas.responses import NextTrialResponse, SubmitTrialResponse
from app.services.participant_service import ParticipantService, ParticipantError


router = APIRouter(prefix="/trials", tags=["trials"])


@router.get("/next", response_model=NextTrialResponse)
def get_next_trial(
    participant: Participant = Depends(get_current_participant),
    session: Session = Depends(get_session),
) -> NextTrialResponse:
    """Return the next non-submitted trial; the participant cannot choose the setup."""
    trial = ParticipantService(session).get_next_trial(participant)
    if trial is None:
        raise HTTPException(status_code=404, detail="Все попытки уже завершены")

    return NextTrialResponse(
        trial_id=trial.id,  # type: ignore[arg-type]
        trial_index=trial.trial_index,
        trials_total=settings.trials_per_participant,
        condition=trial.condition.value,
        obstacle_mode=trial.condition.obstacle_mode,
        window_size=trial.condition.window_size,
        timestamp=trial.timestamp.ts_key,
    )


@router.post("/{trial_id}/start", response_model=NextTrialResponse)
def start_trial(
    trial_id: int,
    participant: Participant = Depends(get_current_participant),
    session: Session = Depends(get_session),
) -> NextTrialResponse:
    """Fix the server-side start time; idempotent (repeated calls do not reset the timer)."""
    service = ParticipantService(session)
    try:
        trial = service.start_trial(participant, trial_id)
    except ParticipantError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message) from None
    return _trial_to_response(trial)


@router.post("/{trial_id}/submit", response_model=SubmitTrialResponse)
def submit_trial(
    trial_id: int,
    request: SubmitTrialRequest,
    participant: Participant = Depends(get_current_participant),
    session: Session = Depends(get_session),
) -> SubmitTrialResponse:
    """Submit raw route data and interaction counters; anti-replay by trial status."""
    service = ParticipantService(session)
    try:
        service.submit_trial(
            participant=participant,
            trial_id=trial_id,
            all_paths=request.all_paths,
            path=request.path,
            num_zoom_actions=request.num_zoom_actions,
            num_forecast_layers_viewed=request.num_forecast_layers_viewed,
        )
    except ParticipantError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message) from None

    submitted = service.submitted_count(participant)
    return SubmitTrialResponse(
        trial_id=trial_id,
        trials_submitted=submitted,
        session_finished=submitted >= settings.trials_per_participant,
    )


def _trial_to_response(trial: Trial) -> NextTrialResponse:
    """Build the response for a started trial."""
    return NextTrialResponse(
        trial_id=trial.id,  # type: ignore[arg-type]
        trial_index=trial.trial_index,
        trials_total=settings.trials_per_participant,
        condition=trial.condition.value,
        obstacle_mode=trial.condition.obstacle_mode,
        window_size=trial.condition.window_size,
        timestamp=trial.timestamp.ts_key,
    )
