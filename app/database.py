"""
Database setup: SQLAlchemy + SQLite.

Using SQLAlchemy Core (engine + session factory) rather than an ORM-heavy
approach keeps things simple and easy to explain in interviews.

The DB file is created automatically at app/support.db on first run.
In the Docker container it lives at /app/support.db (ephemeral, but fine
for a portfolio project — for production you'd use a persistent volume or
a managed Postgres instance).
"""

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# Allow override via env var so tests can use an in-memory DB
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./support.db")

# connect_args is required for SQLite to allow multi-threaded access
engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DB_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: yields a DB session and closes it when done."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
