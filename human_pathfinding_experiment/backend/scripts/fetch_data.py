import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from app.config import settings

READY_MARKERS: tuple[Path, ...] = (
    settings.config_path / "timestamps.pkl",
    settings.config_path / "ab_points.pkl",
)


def dataset_ready(data_path: Path) -> bool:
    """Check whether the dataset and config pickles are already in place."""
    if not all(marker.exists() for marker in READY_MARKERS):
        return False
    return data_path.exists() and any(d.is_dir() for d in data_path.iterdir())


def fetch_archive(url: str, target_dir: Path) -> Path:
    """Download the dataset archive to a temporary file."""
    print(f"Downloading dataset archive from {url} ...")
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".tar.gz")
    tmp_path = Path(tmp_name)
    with urllib.request.urlopen(url) as response, open(tmp_fd, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
            print(".", end="", flush=True)
    print(" done")
    return tmp_path


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    """Extract the dataset archive into the application directory."""
    print(f"Extracting dataset to {target_dir} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir, filter="data")


def main() -> int:
    """Fetch and unpack the dataset archive unless the dataset is already present."""
    if not settings.data_archive_url:
        # Local docker-compose mounts the dataset from the host; nothing to do
        return 0

    if dataset_ready(settings.data_path):
        print("Dataset already present, skipping download")
        return 0

    archive = fetch_archive(settings.data_archive_url, settings.data_path.parent)
    try:
        extract_archive(archive, settings.data_path.parent)
    finally:
        archive.unlink(missing_ok=True)
    print("Dataset ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
