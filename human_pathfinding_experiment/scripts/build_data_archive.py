import argparse
import sys
import tarfile
from pathlib import Path

# Canonical experiment timestamps (single source of truth, shared with the DB seed script)
TIMESTAMPS_FILE = Path(__file__).resolve().parents[1] / "backend" / "scripts" / "experiment_timestamps.txt"


def experiment_timestamps(path: Path = TIMESTAMPS_FILE) -> list[str]:
    """Read the experiment timestamp list (one per line)."""
    with open(path, encoding="utf-8") as file:
        return sorted({line.strip() for line in file if line.strip()})


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

    timestamps = experiment_timestamps()
    build_archive(timestamps, repo_root / "data", repo_root / "config", args.output)

    size_mb = args.output.stat().st_size / 1e6
    print(f"Archive: {args.output} ({size_mb:.1f} MB, {len(timestamps)} timestamps)")
    print("Upload it to Supabase Storage (or any HTTPS host) and set DATA_ARCHIVE_URL for the backend")
    return 0


if __name__ == "__main__":
    sys.exit(main())
