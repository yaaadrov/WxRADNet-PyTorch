import argparse
import shutil
import sys
from pathlib import Path

# Source pieces copied from the backend into the Space build context
COPY_ITEMS: list[str] = ["app", "alembic", "alembic.ini", "pyproject.toml", "scripts"]


def prepare_space_repo(backend_dir: Path, template_dir: Path, output_dir: Path) -> None:
    """Assemble the Hugging Face Space repository: template + backend sources."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for item in COPY_ITEMS:
        source = backend_dir / item
        target = output_dir / item
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(source, target)

    for item in ("Dockerfile", "README.md"):
        shutil.copy2(template_dir / item, output_dir / item)

    # The seed list must ship with the Space (scripts/experiment_timestamps.txt is included
    # via scripts/); nothing else is needed — the dataset is fetched at container start
    # from DATA_ARCHIVE_URL and secrets are configured in the Space settings.


def main() -> int:
    """Build deploy/hf_space and print the git commands to publish it."""
    base = Path(__file__).resolve().parents[1]
    backend_dir = base / "backend"
    template_dir = base / "hf_space_template"

    parser = argparse.ArgumentParser(description="Prepare the Hugging Face Space repository")
    parser.add_argument("--output", type=Path, default=base / "deploy" / "hf_space")
    args = parser.parse_args()

    prepare_space_repo(backend_dir, template_dir, args.output)
    print(f"Space repository assembled: {args.output}")
    print("Publish it (see DEPLOY_GUIDE.md, step 'Backend on Hugging Face Spaces'):")
    print(f"  cd {args.output}")
    print("  git init -b main && git add . && git commit -m 'Backend for the experiment'")
    print("  git remote add space https://huggingface.co/spaces/<USER>/<SPACE>")
    print("  git push https://<USER>:<HF_TOKEN>@huggingface.co/spaces/<USER>/<SPACE> main")
    return 0


if __name__ == "__main__":
    sys.exit(main())
