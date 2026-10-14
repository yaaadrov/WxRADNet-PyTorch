import argparse
import csv
import sys
import tarfile
from pathlib import Path

# Extra timestamp used by the frontend ДЕМО scenarios (C0 case), not part of ranking.csv
DEMO_TIMESTAMP: str = "2023_09_20_13_00_00"


def selected_timestamps(ranking_path: Path) -> list[str]:
    """Read the selected experiment timestamps from the ranking CSV (§2.1)."""
    with open(ranking_path, newline="") as f:
        return sorted(
            row["timestamp"] for row in csv.DictReader(f) if row["selected"] == "True"
        )


def build_archive(timestamps: list[str], data_dir: Path, config_dir: Path, output_path: Path) -> None:
    """Pack the timestamp CSV directories and config pickles into a gzipped tar."""
    missing = [ts for ts in timestamps if not (data_dir / ts).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing timestamp directories: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as tar:
        for ts in timestamps:
            for csv_file in sorted((data_dir / ts).glob("*.csv")):
                # "data/" prefix matches the container layout (DATA_PATH=/app/data)
                tar.add(csv_file, arcname=f"data/{csv_file.relative_to(data_dir)}")
        for pkl in sorted(config_dir.glob("*.pkl")):
            tar.add(pkl, arcname=f"config/{pkl.name}")


def main() -> int:
    """Build the deployable dataset archive for the backend container startup fetch."""
    base = Path(__file__).resolve().parents[1]
    repo_root = base.parent

    parser = argparse.ArgumentParser(description="Build data.tar.gz for the backend DATA_ARCHIVE_URL")
    parser.add_argument("--output", type=Path, default=base / "deploy" / "data.tar.gz")
    args = parser.parse_args()

    timestamps = selected_timestamps(base / "timestamp_selection" / "ranking.csv")
    if DEMO_TIMESTAMP not in timestamps:
        timestamps.append(DEMO_TIMESTAMP)

    build_archive(timestamps, repo_root / "data", repo_root / "config", args.output)

    size_mb = args.output.stat().st_size / 1e6
    print(f"Archive: {args.output} ({size_mb:.1f} MB, {len(timestamps)} timestamps)")
    print("Upload it to Supabase Storage (or any HTTPS host) and set DATA_ARCHIVE_URL for the backend")
    return 0


if __name__ == "__main__":
    sys.exit(main())
