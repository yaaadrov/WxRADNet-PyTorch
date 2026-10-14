import csv
import itertools
import os
from pathlib import Path

from sqlmodel import Session, func, select

from app.db.engine import engine
from app.db.models import AssignmentScheme, ExperimentTimestamp
from app.schemas.enums import ExperimentCondition, TimestampBlock


# One-time seeding script: fills assignment_schemes (Latin square, §1.2) and timestamps (§2.1).
# RANKING_CSV_PATH env override lets the Docker image point at the baked-in CSV copy.
_DEFAULT_RANKING_PATH = Path(__file__).resolve().parents[2] / "timestamp_selection" / "ranking.csv"
RANKING_CSV_PATH = Path(os.environ.get("RANKING_CSV_PATH", _DEFAULT_RANKING_PATH))
EXPECTED_SCHEME_COUNT = 18

# All six condition presentation orders (permutations of C0, C1, C2)
ORDER_PERMUTATIONS = list(itertools.permutations(ExperimentCondition))

# Three block-to-condition mappings: (block_a, block_b, block_c); cyclic shifts of (C0, C1, C2)
BLOCK_MAPPINGS = [
    (ExperimentCondition.C0, ExperimentCondition.C1, ExperimentCondition.C2),
    (ExperimentCondition.C1, ExperimentCondition.C2, ExperimentCondition.C0),
    (ExperimentCondition.C2, ExperimentCondition.C0, ExperimentCondition.C1),
]

BLOCK_CYCLE = [TimestampBlock.A, TimestampBlock.B, TimestampBlock.C]


class SchemeGenerator:
    """Generates the 18 Latin square assignment schemes (6 orders x 3 block mappings)."""

    @classmethod
    def generate(cls) -> list[AssignmentScheme]:
        """Build all unique session schemes."""
        return [
            AssignmentScheme(
                block_a_condition=block_mapping[0],
                block_b_condition=block_mapping[1],
                block_c_condition=block_mapping[2],
                order_1=order[0],
                order_2=order[1],
                order_3=order[2],
            )
            for order, block_mapping in itertools.product(ORDER_PERMUTATIONS, BLOCK_MAPPINGS)
        ]


class TimestampLoader:
    """Loads the selected timestamps from the ranking CSV, splitting them into blocks A/B/C."""

    def __init__(self, ranking_csv_path: Path) -> None:
        """Read the ranking CSV and keep selected timestamps ordered by complexity rank."""
        self._rows = self._read_selected(ranking_csv_path)

    @staticmethod
    def _read_selected(ranking_csv_path: Path) -> list[dict[str, str]]:
        """Read selected=true rows sorted by rank."""
        with open(ranking_csv_path, encoding="utf-8") as file:
            rows = [row for row in csv.DictReader(file) if row["selected"] == "True"]
        return sorted(rows, key=lambda row: int(row["rank"]))

    def load(self, session: Session) -> list[ExperimentTimestamp]:
        """Upsert timestamp rows; block is assigned round-robin by rank (balanced tiers)."""
        existing = {ts.ts_key: ts for ts in session.exec(select(ExperimentTimestamp))}
        result = []
        for row in self._rows:
            block = BLOCK_CYCLE[(int(row["rank"]) - 1) % len(BLOCK_CYCLE)]
            ts = existing.get(row["timestamp"])
            if ts is None:
                ts = ExperimentTimestamp(ts_key=row["timestamp"], block=block)
                session.add(ts)
            else:
                ts.block = block
            result.append(ts)
        return result


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

        timestamps = TimestampLoader(RANKING_CSV_PATH).load(session)
        session.commit()
        for ts in timestamps:
            print(f"timestamp: {ts.ts_key} -> block {ts.block.value}")

        counts = (
            session.exec(select(ExperimentTimestamp.block, func.count()).group_by(ExperimentTimestamp.block)).all()
        )
        print("Block sizes:", {block.value: count for block, count in counts})


if __name__ == "__main__":
    seed()
