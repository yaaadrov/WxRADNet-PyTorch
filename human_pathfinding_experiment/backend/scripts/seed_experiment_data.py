import itertools
import random
from pathlib import Path

from sqlmodel import Session, func, select

from app.db.engine import engine
from app.db.models import AssignmentScheme, ExperimentTimestamp, TimestampBlockMember
from app.schemas.enums import ExperimentCondition, TimestampBlock


# One-time seeding script: fills assignment_schemes (condition orders) and timestamps
# partitioned into timestamp triples (blocks).
_DEFAULT_TIMESTAMPS_PATH = Path(__file__).resolve().parent / "experiment_timestamps.txt"
TIMESTAMPS_PATH = Path(_DEFAULT_TIMESTAMPS_PATH)
EXPECTED_SCHEME_COUNT = 6  # one per condition presentation order

# Fixed shuffle seed: the script re-runs on every container start, and the triple layout
# must stay identical across restarts (a mid-experiment reshuffle would corrupt balance)
TRIPLE_SHUFFLE_SEED = 20261015

# All six condition presentation orders (permutations of C0, C1, C2)
ORDER_PERMUTATIONS = list(itertools.permutations(ExperimentCondition))


def load_experiment_timestamps(path: Path = TIMESTAMPS_PATH) -> list[str]:
    """Read the canonical experiment timestamp list (one per line)."""
    with open(path, encoding="utf-8") as file:
        return sorted({line.strip() for line in file if line.strip()})


class SchemeGenerator:
    """Generates the 6 assignment schemes: one per condition presentation order."""

    @classmethod
    def generate(cls) -> list[AssignmentScheme]:
        """Build all session schemes (timestamps come from the assigned triple, not the scheme)."""
        return [
            AssignmentScheme(order_1=order[0], order_2=order[1], order_3=order[2])
            for order in ORDER_PERMUTATIONS
        ]


class TimestampLoader:
    """Loads experiment timestamps partitioned into triples (blocks A..K).

    Each triple holds 3 consecutive timestamps of one random permutation (cyclically),
    so every timestamp belongs to exactly 3 triples and triples never contain duplicates.
    """

    def __init__(self, timestamps: list[str], rng: random.Random | None = None) -> None:
        """Store the canonical timestamp list and the shuffled triple construction."""
        if len(timestamps) != len(TimestampBlock):
            raise SystemExit(
                f"Expected {len(TimestampBlock)} timestamps (one per triple label), got {len(timestamps)}"
            )
        shuffled = list(timestamps)
        (rng or random.Random(TRIPLE_SHUFFLE_SEED)).shuffle(shuffled)
        self._shuffled = shuffled

    def triples(self) -> dict[TimestampBlock, list[str]]:
        """Map each block label to its 3 timestamp keys."""
        n = len(self._shuffled)
        return {
            block: [self._shuffled[j % n], self._shuffled[(j + 1) % n], self._shuffled[(j + 2) % n]]
            for j, block in enumerate(TimestampBlock)
        }

    def load(self, session: Session) -> None:
        """Upsert timestamps and their triple memberships; idempotent (deterministic triples)."""
        existing = {ts.ts_key: ts for ts in session.exec(select(ExperimentTimestamp))}
        memberships = {
            (member.block, member.timestamp_id)
            for member in session.exec(select(TimestampBlockMember)).all()
        }
        for block, keys in self.triples().items():
            for ts_key in keys:
                ts = existing.get(ts_key)
                if ts is None:
                    ts = ExperimentTimestamp(ts_key=ts_key)
                    session.add(ts)
                    session.flush()
                    existing[ts_key] = ts
                if (block, ts.id) not in memberships:
                    session.add(TimestampBlockMember(block=block, timestamp_id=ts.id))  # type: ignore[arg-type]
                    memberships.add((block, ts.id))  # type: ignore[arg-type]


def seed() -> None:
    """Seed assignment schemes and timestamps; idempotent (safe to re-run)."""
    with Session(engine) as session:
        scheme_count = session.exec(select(func.count()).select_from(AssignmentScheme)).one()
        if scheme_count == EXPECTED_SCHEME_COUNT:
            print(f"assignment_schemes already seeded ({scheme_count} rows), skipping")
        elif scheme_count == 0:
            session.add_all(SchemeGenerator.generate())
            print(f"Seeded {EXPECTED_SCHEME_COUNT} assignment schemes")
        else:
            raise SystemExit(f"assignment_schemes in unexpected state: {scheme_count} rows, expected 0 or {EXPECTED_SCHEME_COUNT}")

        TimestampLoader(load_experiment_timestamps()).load(session)
        session.commit()
        memberships = session.exec(select(TimestampBlockMember)).all()
        ts_keys = {ts.id: ts.ts_key for ts in session.exec(select(ExperimentTimestamp)).all()}
        for member in memberships:
            print(f"block {member.block.value}: {ts_keys[member.timestamp_id]}")

        counts = (
            session.exec(
                select(TimestampBlockMember.timestamp_id, func.count()).group_by(TimestampBlockMember.timestamp_id)
            ).all()
        )
        print("Triples per timestamp:", dict(counts))


if __name__ == "__main__":
    seed()
