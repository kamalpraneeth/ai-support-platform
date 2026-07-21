"""
FastAPI Application — AI-Powered Customer Support Platform

Endpoints:
  GET  /health        — health check
  GET  /              — serves the frontend HTML
  POST /ticket        — classify, score, and store a support ticket
  POST /ticket/reply  — generate an AI draft reply for a ticket
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models import Ticket
from app.ml.classifier import load_model, predict_category
from app.ml.urgency import score_urgency
from app.ml.sentiment import analyze_sentiment
from app.ai_reply import generate_reply

# Load .env file (if present) — no-op if not found
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the classifier once at startup (avoids disk I/O per request)
_classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once at startup, release at shutdown."""
    global _classifier
    # Create DB tables (idempotent — safe to run on every startup)
    Base.metadata.create_all(bind=engine)
    logger.info("Loading classifier model ...")
    _classifier = load_model()
    logger.info("Classifier loaded. App ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="AI-Powered Customer Support Platform",
    description=(
        "Classifies support tickets by category, urgency, and sentiment, "
        "then generates AI-drafted replies using Groq (llama-3.1-8b-instant)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS for iframe embedding and cross-origin API calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend template
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TicketRequest(BaseModel):
    text: str = Field(..., min_length=5, description="The support ticket text")


class TicketResponse(BaseModel):
    id: int
    text: str
    category: str
    urgency: str
    sentiment: str
    message: str = "Ticket received and classified."


class ReplyRequest(BaseModel):
    ticket_id: int = Field(..., description="ID of a previously submitted ticket")


class ReplyResponse(BaseModel):
    ticket_id: int
    reply: str
    is_ai_generated: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """Basic health check — used by Render and Docker HEALTHCHECK."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
def serve_frontend(request: Request):
    """Serve the single-page frontend."""
    return templates.TemplateResponse(request, "index.html")


@app.post("/ticket", response_model=TicketResponse, tags=["Tickets"])
def submit_ticket(payload: TicketRequest, db: Session = Depends(get_db)):
    """
    Submit a support ticket for classification.

    - **text**: The raw support ticket text (min 5 characters).

    Returns the predicted **category**, **urgency**, and **sentiment**,
    and stores the ticket in SQLite.
    """
    text = payload.text.strip()

    # Run ML predictions (cast to str — sklearn returns numpy.str_ types)
    category = str(predict_category(text, model=_classifier))
    urgency = score_urgency(text)
    sentiment = analyze_sentiment(text)

    # Persist to DB
    ticket = Ticket(
        text=text,
        category=category,
        urgency=urgency,
        sentiment=sentiment,
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    logger.info(
        "Ticket #%d: category=%s urgency=%s sentiment=%s",
        ticket.id, category, urgency, sentiment,
    )

    return TicketResponse(
        id=ticket.id,
        text=ticket.text,
        category=category,
        urgency=urgency,
        sentiment=sentiment,
    )


@app.post("/ticket/reply", response_model=ReplyResponse, tags=["Tickets"])
def get_ticket_reply(payload: ReplyRequest, db: Session = Depends(get_db)):
    """
    Generate an AI-drafted reply for a previously submitted ticket.

    - **ticket_id**: The ID returned by POST /ticket.

    Uses Groq (llama-3.1-8b-instant). Falls back to a static professional
    template if the API key is not set or the call fails.
    """
    ticket = db.query(Ticket).filter(Ticket.id == payload.ticket_id).first()
    if not ticket:
        raise HTTPException(
            status_code=404,
            detail=f"Ticket with id={payload.ticket_id} not found.",
        )

    reply, is_ai = generate_reply(ticket.text)

    # Persist the reply back to the ticket record
    ticket.reply = reply
    ticket.is_ai_reply = is_ai
    db.commit()

    return ReplyResponse(
        ticket_id=ticket.id,
        reply=reply,
        is_ai_generated=is_ai,
    )
