from enum import StrEnum


class ExperimentCondition(StrEnum):
    """Experimental condition: cumulative information level shown to the participant.

    C0 = storm borders only (baseline), C1 = borders + forbidden zones (hull),
    C2 = borders + forbidden zones + forecast (target system).
    """

    C0 = "C0"
    C1 = "C1"
    C2 = "C2"

    @property
    def obstacle_mode(self) -> str:
        """Obstacle mode parameter of the condition."""
        match self:
            case ExperimentCondition.C0:
                return "obstacles"
            case ExperimentCondition.C1 | ExperimentCondition.C2:
                return "hull"

    @property
    def window_size(self) -> int:
        """Forecast window size parameter of the condition."""
        match self:
            case ExperimentCondition.C0 | ExperimentCondition.C1:
                return 1
            case ExperimentCondition.C2:
                return 7


class TimestampBlock(StrEnum):
    """Latin square block of timestamps: A, B or C (3 timestamps per block)."""

    A = "A"
    B = "B"
    C = "C"


class TrialStatus(StrEnum):
    """Lifecycle status of a trial: assigned -> in_progress -> submitted."""

    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"

    @property
    def is_terminal(self) -> bool:
        """Whether the status is final and no further transitions are allowed."""
        match self:
            case TrialStatus.SUBMITTED:
                return True
            case TrialStatus.ASSIGNED | TrialStatus.IN_PROGRESS:
                return False
