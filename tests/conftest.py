"""
conftest.py — shared pytest fixtures.

The fundamental problem with in-memory SQLite + FastAPI TestClient:
  - `test_classifier.py` imports app.ml.* at module level, which brings
    app/ onto sys.modules before conftest can set DATABASE_URL.
  - Even if we set DATABASE_URL first, the lifespan's create_all uses the
    engine bound at import time.

Solution:
  Use a real temp file DB for tests (not in-memory), override the `get_db`
  FastAPI dependency to use a session factory bound to that test engine,
  and call `Base.metadata.create_all` on the test engine directly.
  This is 100% isolated from support.db and reset between sessions.
"""

import os
import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Point main.py away from the production DB file
os.environ["DATABASE_URL"] = "sqlite:///./test_support.db"


@pytest.fixture(scope="session")
def client():
    """
    Session-scoped TestClient with:
      - A dedicated test SQLite file (test_support.db, cleaned up after)
      - FastAPI dependency override for get_db → test session
      - Lifespan triggered (loads the ML model)
    """
    from fastapi.testclient import TestClient
    from app.database import Base, get_db
    from app.main import app

    # Build a test engine bound to the env-var URL
    test_engine = create_engine(
        "sqlite:///./test_support.db",
        connect_args={"check_same_thread": False},
    )
    TestSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )

    # Create all tables on the test engine
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Override the DB dependency so all routes use the test engine
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    # Cleanup: drop tables and remove the test DB file
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()
    try:
        if os.path.exists("test_support.db"):
            os.remove("test_support.db")
    except PermissionError:
        pass  # Windows may hold a file lock; file is overwritten on next run
