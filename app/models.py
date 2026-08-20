"""
SQLAlchemy ORM model for the Ticket table.

Columns added in v2.0.0 upgrade:
  - ml_confidence    — classifier confidence score (predict_proba max)
  - rag_chunks_used  — number of knowledge base chunks retrieved per reply
  - llm_latency_ms   — LLM response time in milliseconds
  - evaluation_score — heuristic quality score from evaluation module
  - escalated        — whether the ticket was flagged for human review
"""

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float

from app.database import Base


class Ticket(Base):
    __tablename__ = "tickets"

    # Core fields (v1.0.0 — preserved)
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

    # New fields (v2.0.0 — nullable for backward compatibility with old
    # records)
    # predict_proba max [0.0, 1.0]
    ml_confidence = Column(Float, nullable=True)
    rag_chunks_used = Column(
        Integer,
        nullable=True,
        default=0)  # KB chunks retrieved
    # LLM response time (ms)
    llm_latency_ms = Column(Float, nullable=True)
    # Heuristic quality score
    evaluation_score = Column(Float, nullable=True)
    escalated = Column(
        Boolean,
        nullable=True,
        default=False)    # Human review flag

    # Computer Vision fields (v2.1.0)
    has_image = Column(Boolean, nullable=True, default=False)
    # JSON string of latency metrics
    cv_metrics = Column(String, nullable=True)
    # JSON string of objects + confidences
    detected_objects = Column(String, nullable=True)


class ChatSession(Base):
    """Represents a multi-turn chatbot conversation session."""
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class ChatMessage(Base):
    """Represents a single message inside a ChatSession."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(Integer, nullable=False, index=True)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
