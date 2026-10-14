from collections.abc import Generator

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from app.config import settings


def create_db_engine(database_url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine for the configured Postgres database."""
    return create_engine(
        database_url or settings.sqlalchemy_database_url, echo=False, pool_pre_ping=True
    )


engine: Engine = create_db_engine()


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a database session."""
    with Session(engine) as session:
        yield session


def init_db() -> None:
    """Create database schema directly from models (dev/test shortcut; use alembic in prod)."""
    from app.db import models  # noqa: F401  (import registers all tables in SQLModel.metadata)

    SQLModel.metadata.create_all(engine)
