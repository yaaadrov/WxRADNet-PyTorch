"""Initial experiment tables: participants, timestamps, assignment schemes, trials.

Revision ID: 0001_initial_tables
Revises:
Create Date: 2026-10-14

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

from app.schemas.enums import ExperimentCondition, TimestampBlock, TrialStatus


# revision identifiers, used by Alembic.
revision: str = "0001_initial_tables"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _condition_type() -> sa.Enum:
    """Build a non-native enum type matching the model definition."""
    return sa.Enum(
        ExperimentCondition,
        native_enum=False,
        values_callable=lambda e: [member.value for member in e],
        length=8,
    )


def _status_type() -> sa.Enum:
    """Build a non-native enum type for trial status."""
    return sa.Enum(
        TrialStatus,
        native_enum=False,
        values_callable=lambda e: [member.value for member in e],
        length=16,
    )


def upgrade() -> None:
    """Create all experiment tables."""
    op.create_table(
        "participants",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("surname", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("group_name", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consented_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_completed", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("ease_rating_borders", sa.Integer(), nullable=True),
        sa.Column("ease_rating_zones", sa.Integer(), nullable=True),
        sa.Column("ease_rating_forecast", sa.Integer(), nullable=True),
        sa.Column("awareness_zones_helps", sa.Boolean(), nullable=True),
        sa.Column("awareness_forecast_helps", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("surname", "name", "group_name", name="uq_participant_identity"),
    )

    op.create_table(
        "timestamps",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("ts_key", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ts_key", name="uq_timestamp_ts_key"),
    )

    op.create_table(
        "timestamp_blocks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "block",
            sa.Enum(TimestampBlock, native_enum=False, values_callable=lambda e: [m.value for m in e], length=2),
            nullable=False,
        ),
        sa.Column("timestamp_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["timestamp_id"], ["timestamps.id"], name="fk_block_timestamp"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("block", "timestamp_id", name="uq_block_timestamp"),
    )

    op.create_table(
        "assignment_schemes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("order_1", _condition_type(), nullable=False),
        sa.Column("order_2", _condition_type(), nullable=False),
        sa.Column("order_3", _condition_type(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "participant_assignment",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("participant_id", sa.Integer(), nullable=False),
        sa.Column("scheme_id", sa.Integer(), nullable=False),
        sa.Column(
            "block",
            sa.Enum(TimestampBlock, native_enum=False, values_callable=lambda e: [m.value for m in e], length=2),
            nullable=False,
        ),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["participant_id"], ["participants.id"], name="fk_assignment_participant"),
        sa.ForeignKeyConstraint(["scheme_id"], ["assignment_schemes.id"], name="fk_assignment_scheme"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("participant_id", name="uq_assignment_participant"),
    )

    op.create_table(
        "trials",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("participant_id", sa.Integer(), nullable=False),
        sa.Column("timestamp_id", sa.Integer(), nullable=False),
        sa.Column("condition", _condition_type(), nullable=False),
        sa.Column("trial_index", sa.Integer(), nullable=False),
        sa.Column("status", _status_type(), nullable=False),
        sa.Column("server_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("server_finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("all_paths", JSONB(), nullable=True),
        sa.Column("path", JSONB(), nullable=True),
        sa.Column("num_zoom_actions", sa.Integer(), nullable=True),
        sa.Column("num_forecast_layers_viewed", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["participant_id"], ["participants.id"], name="fk_trial_participant"),
        sa.ForeignKeyConstraint(["timestamp_id"], ["timestamps.id"], name="fk_trial_timestamp"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_trials_participant_id"), "trials", ["participant_id"], unique=False)


def downgrade() -> None:
    """Drop all experiment tables in reverse dependency order."""
    op.drop_index(op.f("ix_trials_participant_id"), table_name="trials")
    op.drop_table("trials")
    op.drop_table("participant_assignment")
    op.drop_table("assignment_schemes")
    op.drop_table("timestamp_blocks")
    op.drop_table("timestamps")
    op.drop_table("participants")
