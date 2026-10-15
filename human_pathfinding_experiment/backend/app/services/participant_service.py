import itertools
import math
import random
from datetime import datetime, timezone

from sqlalchemy import func
from sqlmodel import Session, select

from app.config import settings
from app.db.models import AssignmentScheme, ExperimentTimestamp, Participant, ParticipantAssignment, TimestampBlockMember, Trial
from app.schemas.enums import ExperimentCondition, TimestampBlock, TrialStatus
from app.schemas.requests import QuestionnaireRequest
from app.services.data_service import DataService


class ParticipantError(Exception):
    """Base class for participant workflow errors (mapped to HTTP codes in routers)."""

    def __init__(self, message: str, status_code: int) -> None:
        """Store the human-readable message and the HTTP status code."""
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ParticipantAlreadyCompletedError(ParticipantError):
    """Raised on login of a participant who already finished the experiment."""

    def __init__(self) -> None:
        """Set the fixed user-facing message."""
        super().__init__("Эксперимент уже пройден", status_code=403)


class TrialAccessError(ParticipantError):
    """Raised when the trial does not belong to the authenticated participant."""

    def __init__(self) -> None:
        """Set the fixed user-facing message."""
        super().__init__("Попытка не найдена", status_code=404)


class TrialAlreadySubmittedError(ParticipantError):
    """Raised on repeated submit of the same trial (anti-replay)."""

    def __init__(self) -> None:
        """Set the fixed user-facing message."""
        super().__init__("Результат уже отправлен", status_code=409)


class TrialNotSubmittedError(ParticipantError):
    """Raised when the questionnaire is requested before all trials are submitted."""

    def __init__(self) -> None:
        """Set the fixed user-facing message."""
        super().__init__("Не все попытки завершены", status_code=403)


class PathDoesNotReachBError(ParticipantError):
    """Raised when the submitted route does not reach point B."""

    def __init__(self) -> None:
        """Set the fixed user-facing message."""
        super().__init__("Маршрут не доходит до точки B", status_code=422)


class ParticipantService:
    """Participant lifecycle: on-the-fly creation, scheme assignment, trials, questionnaire."""

    def __init__(self, session: Session) -> None:
        """Store the request-scoped database session."""
        self._session = session
        self._data_service = DataService()

    # ------------------------------------------------------------------ login

    def login(self, surname: str, name: str, group: str) -> Participant:
        """Return the existing incomplete participant or create a new one on the fly."""
        participant = self._session.exec(
            select(Participant).where(
                Participant.surname == surname,
                Participant.name == name,
                Participant.group_name == group,
            )
        ).first()

        if participant is not None:
            if participant.is_completed:
                raise ParticipantAlreadyCompletedError()
            return participant

        participant = Participant(surname=surname, name=name, group_name=group, consented_at=utc_now())
        self._session.add(participant)
        self._session.flush()

        scheme = self._pick_least_used_scheme()
        self._assign_scheme(participant, scheme)
        self._session.commit()
        self._session.refresh(participant)
        return participant

    # ------------------------------------------------------- trial lifecycle

    def get_next_trial(self, participant: Participant) -> Trial | None:
        """Return the next non-submitted trial (lowest index first), if any."""
        statement = (
            select(Trial)
            .where(Trial.participant_id == participant.id, Trial.status != TrialStatus.SUBMITTED)
            .order_by(Trial.trial_index)
        )
        return self._session.exec(statement).first()

    def get_trial(self, participant: Participant, trial_id: int) -> Trial:
        """Load a trial owned by the participant or raise TrialAccessError."""
        trial = self._session.get(Trial, trial_id)
        if trial is None or trial.participant_id != participant.id:
            raise TrialAccessError()
        return trial

    def start_trial(self, participant: Participant, trial_id: int) -> Trial:
        """Fix the server-side start time; idempotent (a restart does not reset the timer)."""
        trial = self.get_trial(participant, trial_id)
        if trial.status == TrialStatus.ASSIGNED:
            trial.status = TrialStatus.IN_PROGRESS
            trial.server_started_at = utc_now()
            self._session.commit()
            self._session.refresh(trial)
        return trial

    def submit_trial(
        self,
        participant: Participant,
        trial_id: int,
        all_paths: list[list[tuple[float, float]]],
        path: list[list[tuple[float, float]]],
        num_zoom_actions: int,
        num_forecast_layers_viewed: int,
    ) -> Trial:
        """Store raw route data and finalize the trial (anti-replay by status)."""
        trial = self.get_trial(participant, trial_id)
        if trial.status == TrialStatus.SUBMITTED:
            raise TrialAlreadySubmittedError()

        self._ensure_path_reaches_b(trial, path)

        trial.all_paths = all_paths
        trial.path = path
        trial.num_zoom_actions = num_zoom_actions
        trial.num_forecast_layers_viewed = num_forecast_layers_viewed
        trial.server_finished_at = utc_now()
        trial.status = TrialStatus.SUBMITTED
        self._session.commit()
        self._session.refresh(trial)
        return trial

    def submitted_count(self, participant: Participant) -> int:
        """Count submitted trials of the participant."""
        return len(
            self._session.exec(
                select(Trial).where(
                    Trial.participant_id == participant.id,
                    Trial.status == TrialStatus.SUBMITTED,
                )
            ).all()
        )

    def save_questionnaire(self, participant: Participant, answers: QuestionnaireRequest) -> None:
        """Store the final questionnaire answers and mark the participant as completed."""
        if self.submitted_count(participant) < settings.trials_per_participant:
            raise TrialNotSubmittedError()

        participant.ease_rating_borders = answers.ease_rating_borders
        participant.ease_rating_zones = answers.ease_rating_zones
        participant.ease_rating_forecast = answers.ease_rating_forecast
        participant.awareness_zones_helps = answers.awareness_zones_helps
        participant.awareness_forecast_helps = answers.awareness_forecast_helps
        participant.is_completed = True
        self._session.commit()

    # --------------------------------------------------------------- helpers

    def _assign_scheme(self, participant: Participant, scheme: AssignmentScheme) -> None:
        """Create the scheme assignment and all trials for the session.

        The participant gets the least-used timestamp triple; the session conditions come
        from the scheme's presentation order and are paired with the triple's timestamps
        by the least-used (timestamp, condition) bijection.
        """
        block = self._pick_least_used_block()
        self._session.add(
            ParticipantAssignment(participant_id=participant.id, scheme_id=scheme.id, block=block)  # type: ignore[arg-type]
        )

        conditions = [getattr(scheme, f"order_{i}") for i in range(1, settings.trials_per_participant + 1)]
        pairing = self._pick_timestamp_pairing(block, conditions)
        for trial_index, condition in enumerate(conditions, start=1):
            self._session.add(
                Trial(
                    participant_id=participant.id,
                    timestamp_id=pairing[condition].id,  # type: ignore[arg-type]
                    condition=condition,
                    trial_index=trial_index,
                    status=TrialStatus.ASSIGNED,
                )
            )

    def _pick_least_used_scheme(self) -> AssignmentScheme:
        """Pick randomly among the schemes assigned to the fewest participants.

        Random tie-break (instead of id order) decorrelates the presentation order
        of conditions between participants arriving one after another.
        """
        usage = {
            row[0]: row[1]
            for row in self._session.exec(
                select(ParticipantAssignment.scheme_id, func.count()).group_by(ParticipantAssignment.scheme_id)
            ).all()
        }
        schemes = self._session.exec(select(AssignmentScheme)).all()
        if not schemes:
            raise ParticipantError("Схемы не загружены", status_code=500)
        min_usage = min(usage.get(s.id, 0) for s in schemes)
        return random.choice([s for s in schemes if usage.get(s.id, 0) == min_usage])

    def _pick_least_used_block(self) -> TimestampBlock:
        """Pick randomly among the timestamp triples assigned to the fewest participants.

        Random tie-break (instead of label order) decorrelates triple selection between
        participants arriving one after another.
        """
        usage = {
            row[0]: row[1]
            for row in self._session.exec(
                select(ParticipantAssignment.block, func.count()).group_by(ParticipantAssignment.block)
            ).all()
        }
        blocks = list(TimestampBlock)
        min_usage = min(usage.get(block, 0) for block in blocks)
        return random.choice([block for block in blocks if usage.get(block, 0) == min_usage])

    def _pick_timestamp_pairing(
        self,
        block: TimestampBlock,
        conditions: list[ExperimentCondition],
    ) -> dict[ExperimentCondition, ExperimentTimestamp]:
        """Pair the session conditions with the participant's triple timestamps.

        Among all bijections of the triple's active timestamps onto the conditions, picks
        the one whose resulting (timestamp, condition) usage cells are closest to the global
        counts (squared-count score, random tie-break) — each timestamp eventually appears
        under each condition roughly equally (§1.2 greedy balancer).
        """
        candidates = self._session.exec(
            select(ExperimentTimestamp)
            .join(TimestampBlockMember, TimestampBlockMember.timestamp_id == ExperimentTimestamp.id)
            .where(
                TimestampBlockMember.block == block,
                ExperimentTimestamp.is_active == True,  # noqa: E712  (SQLModel comparison)
            )
        ).all()
        if len(candidates) != len(conditions):
            raise ParticipantError(
                f"В тройке {block.value} недостаточно активных timestamp'ов", status_code=500
            )

        usage = {
            (row[0], row[1]): row[2]
            for row in self._session.exec(
                select(Trial.timestamp_id, Trial.condition, func.count()).group_by(
                    Trial.timestamp_id, Trial.condition
                )
            ).all()
        }

        def score(pairing: tuple[ExperimentTimestamp, ...]) -> int:
            return sum((usage.get((ts.id, condition), 0) + 1) ** 2 for condition, ts in zip(conditions, pairing))

        best_score = min(score(permutation) for permutation in itertools.permutations(candidates))
        best_pairings = [
            permutation
            for permutation in itertools.permutations(candidates)
            if score(permutation) == best_score
        ]
        return dict(zip(conditions, random.choice(best_pairings)))

    def _ensure_path_reaches_b(self, trial: Trial, path: list[list[tuple[float, float]]]) -> None:
        """Minimal sanity check: the final route must reach point B (§3.2)."""
        if not path or not path[-1]:
            raise PathDoesNotReachBError()

        a_point, b_point = self._data_service.get_ab_points_for_timestamp(trial.timestamp.ts_key)
        last_point = path[-1][-1]
        distance_to_b = math.hypot(last_point[0] - b_point[0], last_point[1] - b_point[1])
        if distance_to_b > settings.snap_distance_m:
            raise PathDoesNotReachBError()


def utc_now() -> datetime:
    """Return the current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)
