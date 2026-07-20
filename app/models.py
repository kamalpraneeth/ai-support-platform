"""
SQLAlchemy ORM model for the Ticket table.

One table keeps this simple and interview-friendly.
All predictions (category, urgency, sentiment) are stored alongside the
raw ticket text so you can inspect the DB with `sqlite3 support.db`.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean

from app.database import Base


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    text = Column(Text, nullable=False)
    category = Column(String(50), nullable=False)
    urgency = Column(String(10), nullable=False)
    sentiment = Column(String(10), nullable=False)
    reply = Column(Text, nullable=True)
    is_ai_reply = Column(Boolean, default=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
