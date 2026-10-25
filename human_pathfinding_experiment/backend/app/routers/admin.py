import csv
from io import StringIO

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from app.db.engine import get_session
from app.db.models import Participant, ParticipantAssignment, Trial
from app.routers.auth import require_admin
from app.schemas.enums import ExperimentCondition, TrialStatus
from app.schemas.responses import (
    AdminSummaryResponse,
    ConditionCount,
    SchemeUsage,
    TrialsByStatus,
)

router = APIRouter(dependencies=[Depends(require_admin)])

EXPORT_COLUMNS: list[str] = [
    "trial_id",
    "participant_id",
    "surname",
    "name",
    "group_name",
    "is_completed",
    "ease_rating_borders",
    "ease_rating_zones",
    "ease_rating_forecast",
    "awareness_zones_helps",
    "awareness_forecast_helps",
    "condition",
    "trial_index",
    "status",
    "ts_key",
    "server_started_at",
    "server_finished_at",
    "num_zoom_actions",
    "num_forecast_layers_viewed",
    "all_paths",
    "path",
]


@router.get("/summary", response_model=AdminSummaryResponse)
def get_summary(session: Session = Depends(get_session)) -> AdminSummaryResponse:
    """Live session progress for the experiment day: participants, trials, scheme balance (§3.4)."""
    participants = session.exec(select(Participant)).all()
    trials = session.exec(select(Trial)).all()
    assignments = session.exec(select(ParticipantAssignment)).all()

    status_counts = {status: sum(trial.status == status for trial in trials) for status in TrialStatus}
    condition_counts = {
        condition: sum(
            trial.condition == condition and trial.status == TrialStatus.SUBMITTED for trial in trials
        )
        for condition in ExperimentCondition
    }
    scheme_usage: dict[int, int] = {}
    for assignment in assignments:
        scheme_usage[assignment.scheme_id] = scheme_usage.get(assignment.scheme_id, 0) + 1

    return AdminSummaryResponse(
        participants_total=len(participants),
        participants_completed=sum(p.is_completed for p in participants),
        trials_by_status=TrialsByStatus(
            assigned=status_counts[TrialStatus.ASSIGNED],
            in_progress=status_counts[TrialStatus.IN_PROGRESS],
            submitted=status_counts[TrialStatus.SUBMITTED],
        ),
        submitted_by_condition=[
            ConditionCount(condition=condition, count=condition_counts[condition])
            for condition in ExperimentCondition
        ],
        scheme_usage=sorted(
            (SchemeUsage(scheme_id=scheme_id, assignments=usage) for scheme_id, usage in scheme_usage.items()),
            key=lambda usage: usage.scheme_id,
        ),
    )


@router.get("/export")
def export_trials(session: Session = Depends(get_session)) -> StreamingResponse:
    """Export all trials with participant identity as CSV for the analysis script (§3.5)."""
    trials = session.exec(
        select(Trial, Participant)
        .join(Participant, Trial.participant_id == Participant.id)
        .order_by(Trial.id)
    ).all()

    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(EXPORT_COLUMNS)
    for trial, participant in trials:
        writer.writerow(
            [
                trial.id,
                participant.id,
                participant.surname,
                participant.name,
                participant.group_name,
                participant.is_completed,
                participant.ease_rating_borders,
                participant.ease_rating_zones,
                participant.ease_rating_forecast,
                participant.awareness_zones_helps,
                participant.awareness_forecast_helps,
                trial.condition.value,
                trial.trial_index,
                trial.status.value,
                trial.timestamp.ts_key if trial.timestamp else None,
                trial.server_started_at.isoformat() if trial.server_started_at else None,
                trial.server_finished_at.isoformat() if trial.server_finished_at else None,
                trial.num_zoom_actions,
                trial.num_forecast_layers_viewed,
                trial.all_paths,
                trial.path,
            ]
        )

    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=trials_export.csv"},
    )
