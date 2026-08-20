"""
Analytics Module: database-backed ticket statistics and AI performance metrics.

Provides:
  - Ticket distribution (category, urgency, sentiment) from the database
  - AI usage stats (LLM calls, fallback rate, avg latency)
  - RAG retrieval statistics
  - Escalation and evaluation stats
  - ML model evaluation metrics (from saved evaluation_report.json)

All stats are computed from real data — no fabricated numbers.
DB-based stats reflect all tickets in the database.
In-memory stats (counters) reflect only the current session.
"""

import logging

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models import Ticket
from app.ml.evaluate import load_evaluation_report
from app.monitoring import counters

logger = logging.getLogger(__name__)


def get_ticket_stats(db: Session) -> dict:
    """
    Compute ticket distribution statistics from the database.

    Queries the tickets table for:
      - Total count
      - Distribution by category, urgency, sentiment
      - AI reply usage stats
      - Average ML confidence (if stored)
    """
    total = db.query(func.count(Ticket.id)).scalar() or 0

    if total == 0:
        return {
            "total_tickets": 0,
            "by_category": {},
            "by_urgency": {},
            "by_sentiment": {},
            "ai_replies": {"total": 0, "ai_generated": 0, "fallback": 0},
        }

    # Category distribution
    category_rows = (
        db.query(Ticket.category, func.count(Ticket.id))
        .group_by(Ticket.category)
        .all()
    )
    by_category = {row[0]: row[1] for row in category_rows}

    # Urgency distribution
    urgency_rows = (
        db.query(Ticket.urgency, func.count(Ticket.id))
        .group_by(Ticket.urgency)
        .all()
    )
    by_urgency = {row[0]: row[1] for row in urgency_rows}

    # Sentiment distribution
    sentiment_rows = (
        db.query(Ticket.sentiment, func.count(Ticket.id))
        .group_by(Ticket.sentiment)
        .all()
    )
    by_sentiment = {row[0]: row[1] for row in sentiment_rows}

    # AI reply stats
    total_with_reply = db.query(
        func.count(
            Ticket.id)).filter(
        Ticket.reply.isnot(None)).scalar() or 0
    ai_generated = db.query(func.count(Ticket.id)).filter(
        Ticket.is_ai_reply.is_(True)).scalar() or 0
    fallback = total_with_reply - ai_generated

    # Average confidence (may be None for old records without this column)
    try:
        avg_confidence_result = db.query(
            func.avg(Ticket.ml_confidence)).scalar()
        avg_confidence = round(
            float(avg_confidence_result),
            4) if avg_confidence_result else None
    except Exception:
        avg_confidence = None

    return {
        "total_tickets": total,
        "by_category": by_category,
        "by_urgency": by_urgency,
        "by_sentiment": by_sentiment,
        "ai_replies": {
            "total": total_with_reply,
            "ai_generated": ai_generated,
            "fallback": fallback,
        },
        "avg_ml_confidence": avg_confidence,
    }


def get_ai_stats() -> dict:
    """
    Return AI performance stats from in-memory counters.

    Note: Resets on application restart.
    For persistent stats, use get_ticket_stats() which queries the database.
    """
    return {
        "session_stats": counters.to_dict(),
        "note": (
            "Session stats reset on application restart. "
            "Database stats via /analytics/tickets are persistent."
        ),
    }


def get_model_metrics() -> dict:
    """
    Return ML model evaluation metrics from the saved evaluation report.

    Returns the data computed during the last training run.
    Returns an error message if the report file is not found.
    """
    report = load_evaluation_report()
    if report is None:
        return {
            "error": "Evaluation report not found. Run `python -m app.ml.train` to generate it.",
            "tip": "The report is created automatically during training.",
        }
    return report


def get_full_analytics(db: Session) -> dict:
    """
    Return a combined analytics payload for the /analytics endpoint.

    Combines:
      - Database-backed ticket stats (persistent)
      - In-memory session counters (resets on restart)
      - Saved ML evaluation metrics
    """
    return {
        "tickets": get_ticket_stats(db),
        "session": counters.to_dict(),
        "ml_evaluation": get_model_metrics(),
    }
