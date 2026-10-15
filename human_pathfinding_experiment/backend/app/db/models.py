from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Optional

from sqlalchemy import Column, DateTime, Enum, UniqueConstraint, false, true
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, SQLModel

from app.schemas.enums import ExperimentCondition, TimestampBlock, TrialStatus


def utc_now() -> datetime:
    """Return the current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


def enum_column(enum_type: type[StrEnum], length: int) -> Column:
    """Build a non-native enum column for the given StrEnum type."""
    return Column(
        Enum(
            enum_type,
            native_enum=False,
            values_callable=lambda e: [member.value for member in e],
            length=length,
        ),
        nullable=False,
    )


def condition_column() -> Column:
    """Build a non-native enum column for ExperimentCondition values."""
    return enum_column(ExperimentCondition, length=8)


def status_column() -> Column:
    """Build a non-native enum column for TrialStatus values."""
    return enum_column(TrialStatus, length=16)


class Participant(SQLModel, table=True):
    """Experiment participant, created on first login (no preloading)."""

    __tablename__ = "participants"

    __table_args__ = (UniqueConstraint("surname", "name", "group_name", name="uq_participant_identity"),)

    id: int | None = Field(default=None, primary_key=True)
    surname: str = Field(max_length=128)
    name: str = Field(max_length=128)
    group_name: str = Field(max_length=128)
    created_at: datetime = Field(default_factory=utc_now, sa_type=DateTime(timezone=True))
    consented_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))
    is_completed: bool = Field(default=False, sa_column_kwargs={"server_default": false()})

    # Final session questionnaire answers (filled after all trials are submitted)
    ease_rating_borders: int | None = Field(default=None, ge=1, le=5)
    ease_rating_zones: int | None = Field(default=None, ge=1, le=5)
    ease_rating_forecast: int | None = Field(default=None, ge=1, le=5)
    awareness_zones_helps: bool | None = Field(default=None)
    awareness_forecast_helps: bool | None = Field(default=None)

    assignment: Optional["ParticipantAssignment"] = Relationship(back_populates="participant")
    trials: list["Trial"] = Relationship(back_populates="participant")


class ExperimentTimestamp(SQLModel, table=True):
    """Storm timestamp available for the experiment.

    Triple membership (each timestamp belongs to exactly 3 triples) is stored in
    TimestampBlockMember. Reference lengths (AI optimal path, straight-line A->B) are
    deliberately NOT stored here: they are only needed post-hoc in the analysis script and
    are read from human_pathfinding_experiment/timestamp_selection/ranking.csv (§1.5/§3.5).
    """

    __tablename__ = "timestamps"

    id: int | None = Field(default=None, primary_key=True)
    ts_key: str = Field(unique=True, max_length=32, description="Timestamp directory name")
    is_active: bool = Field(default=True, sa_column_kwargs={"server_default": true()})

    trials: list["Trial"] = Relationship(back_populates="timestamp")
    block_memberships: list["TimestampBlockMember"] = Relationship(back_populates="timestamp")


class TimestampBlockMember(SQLModel, table=True):
    """Membership of a timestamp in a timestamp triple: each timestamp belongs to 3 triples."""

    __tablename__ = "timestamp_blocks"

    __table_args__ = (UniqueConstraint("block", "timestamp_id", name="uq_block_timestamp"),)

    id: int | None = Field(default=None, primary_key=True)
    block: TimestampBlock = Field(
        sa_column=enum_column(TimestampBlock, length=2),
        description="Label of the timestamp triple (A..K, 11 triples of 3 timestamps)",
    )
    timestamp_id: int = Field(foreign_key="timestamps.id")

    timestamp: ExperimentTimestamp = Relationship(back_populates="block_memberships")


class AssignmentScheme(SQLModel, table=True):
    """Condition presentation order: a permutation of C0/C1/C2 over the session's trials.

    Timestamps are no longer bound to the scheme: they come from the participant's
    assigned timestamp triple (see ParticipantAssignment.block).
    """

    __tablename__ = "assignment_schemes"

    id: int | None = Field(default=None, primary_key=True)
    order_1: ExperimentCondition = Field(sa_column=condition_column())
    order_2: ExperimentCondition = Field(sa_column=condition_column())
    order_3: ExperimentCondition = Field(sa_column=condition_column())

    assignments: list["ParticipantAssignment"] = Relationship(back_populates="scheme")


class ParticipantAssignment(SQLModel, table=True):
    """Assignment of a condition order and a timestamp triple to a participant.

    Both are chosen as least-used at creation time.
    """

    __tablename__ = "participant_assignment"

    id: int | None = Field(default=None, primary_key=True)
    participant_id: int = Field(foreign_key="participants.id", unique=True)
    scheme_id: int = Field(foreign_key="assignment_schemes.id")
    block: TimestampBlock = Field(
        sa_column=enum_column(TimestampBlock, length=2),
        description="Timestamp triple assigned to the participant",
    )
    assigned_at: datetime = Field(default_factory=utc_now, sa_type=DateTime(timezone=True))

    participant: Participant = Relationship(back_populates="assignment")
    scheme: AssignmentScheme = Relationship(back_populates="assignments")


class Trial(SQLModel, table=True):
    """Single attempt of a participant: raw route data plus interaction counters only.

    All derived metrics (excess_ratio, success, min_clearance_m, ...) are computed
    post-hoc in the analysis script, never stored here.
    """

    __tablename__ = "trials"

    id: int | None = Field(default=None, primary_key=True)
    participant_id: int = Field(foreign_key="participants.id", index=True)
    timestamp_id: int = Field(foreign_key="timestamps.id")
    condition: ExperimentCondition = Field(sa_column=condition_column())
    trial_index: int = Field(ge=1, le=3, description="Position within the participant's session (1..3)")
    status: TrialStatus = Field(
        default=TrialStatus.ASSIGNED,
        sa_column=status_column(),
    )

    server_started_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))
    server_finished_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))

    all_paths: list[list[tuple[float, float]]] | None = Field(default=None, sa_column=Column(JSONB))
    path: list[list[tuple[float, float]]] | None = Field(default=None, sa_column=Column(JSONB))
    num_zoom_actions: int | None = Field(default=None)
    num_forecast_layers_viewed: int | None = Field(default=None)

    participant: Participant = Relationship(back_populates="trials")
    timestamp: ExperimentTimestamp = Relationship(back_populates="trials")

    @property
    def raw_route_payload(self) -> dict[str, Any]:
        """Raw route data as submitted by the frontend."""
        return {"all_paths": self.all_paths, "path": self.path}
