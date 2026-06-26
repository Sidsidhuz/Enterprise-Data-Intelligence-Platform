"""
app/database.py
================

SQLAlchemy engine, session factory, and declarative base for the SQLite
database. This is intentionally simple: there's a single SQLite file, a
single engine, and a single way to get a session — appropriate for a local,
single-user application.

Usage in a route or script:

    from app.database import get_db

    def some_function(db: Session = Depends(get_db)):
        db.query(Dataset).all()

To create the tables for the first time (or after adding new models):

    from app.database import init_db
    init_db()
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

# `check_same_thread=False` is required for SQLite when the same connection
# pool is shared across FastAPI's request-handling threads. This is safe here
# because SQLAlchemy's default session-per-request pattern still ensures each
# request gets its own Session object.
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all ORM models. Import this from app/models/*.py files."""
    pass


def init_db() -> None:
    """
    Create all tables defined by models that inherit from `Base`.

    Safe to call multiple times — SQLAlchemy only creates tables that don't
    already exist. Must be called *after* all model modules have been
    imported (see app/models/__init__.py), otherwise their tables won't be
    registered on `Base.metadata` yet.
    """
    # Ensure local data folders exist before anything tries to write to them.
    settings.ensure_data_dirs_exist()

    # Import models here (not at module load time) to avoid circular imports
    # between database.py and the model modules, which themselves import Base
    # from this file.
    import app.models  # noqa: F401  (import triggers model registration)

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session and guarantees it is
    closed after the request finishes, even if an exception is raised.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Context manager for using a database session *outside* of a FastAPI
    request (e.g. in a background task, a script, or a test). Commits on
    success, rolls back on error, and always closes the session.

        with session_scope() as db:
            db.add(some_object)
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
