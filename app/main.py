"""
FastAPI Application — AI-Powered Customer Support Platform v2.0.0

Endpoints:
  GET  /health              — health check (extended with component status)
  GET  /                    — serves the frontend HTML
  POST /ticket              — classify, score, and store a support ticket
  POST /ticket/reply        — full RAG + LLM orchestration for a ticket reply
  GET  /analytics           — full analytics dashboard
  GET  /analytics/tickets   — ticket distribution stats
  GET  /analytics/ai        — AI/LLM session performance stats
  GET  /metrics             — ML model evaluation metrics

Changes from v1.0.0:
  - /ticket now returns ml_confidence score
  - /ticket/reply now uses the full AI orchestration pipeline (RAG + prompts + eval)
  - Added /analytics/* endpoints
  - Added /metrics endpoint
  - ML classifier loaded with confidence scoring
  - RAG KnowledgeRetriever initialized at startup
  - Monitoring module wired into all endpoints
  - Responsible AI input validation on /ticket
  - CORS origin configurable via ALLOWED_ORIGINS env var
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models import Ticket
from app.ml.classifier import load_model, predict_with_confidence
from app.ml.urgency import score_urgency
from app.ml.sentiment import analyze_sentiment
from app.ai_orchestrator import orchestrate
from app.rag.knowledge_base import load_knowledge_base
from app.rag.retriever import KnowledgeRetriever
from app.analytics import get_ticket_stats, get_ai_stats, get_model_metrics, get_full_analytics
from app.monitoring import log_ml_event, log_api_request, get_health_status, counters
from app.cv import validate_image_file, load_and_preprocess_image, cv_detector, ImageValidationError

# Load .env file (if present) — no-op if not found
load_dotenv()

# ServiceNow Configuration settings
SERVICENOW_INSTANCE = os.getenv("SERVICENOW_INSTANCE")
SERVICENOW_USER = os.getenv("SERVICENOW_USER")
SERVICENOW_PASS = os.getenv("SERVICENOW_PASS")
SERVICENOW_SCOPE = os.getenv("SERVICENOW_SCOPE", "x_2179494_ai_inc_0")
SERVICENOW_API_ID = os.getenv("SERVICENOW_API_ID", "incident_intake")

# CORS: configurable via env var, defaults to all origins (dev-friendly)
# In production, set ALLOWED_ORIGINS=https://yourdomain.com
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state (loaded once at startup)
# ---------------------------------------------------------------------------

_classifier = None
_retriever: KnowledgeRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once at startup, release at shutdown."""
    global _classifier, _retriever

    # Create DB tables (idempotent — safe to run on every startup)
    Base.metadata.create_all(bind=engine)

    # Load ML classifier
    logger.info("Loading classifier model ...")
    _classifier = load_model()
    logger.info("Classifier loaded.")

    # Load knowledge base and build RAG retriever index
    logger.info("Loading knowledge base ...")
    documents = load_knowledge_base()
    _retriever = KnowledgeRetriever()
    _retriever.build_index(documents)
    logger.info("RAG retriever ready: %d documents indexed.", _retriever.document_count)

    logger.info("App ready. v2.0.0")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="AI-Powered Customer Support Platform",
    description=(
        "Classifies support tickets by category, urgency, and sentiment. "
        "Generates RAG-augmented AI replies using Groq (llama-3.1-8b-instant). "
        "Includes ML evaluation, response quality assessment, responsible AI checks, "
        "and analytics endpoints."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
    ml_confidence: float
    escalated: bool
    message: str = "Ticket received and classified."


class ReplyRequest(BaseModel):
    ticket_id: int = Field(..., description="ID of a previously submitted ticket")


class ReplyResponse(BaseModel):
    ticket_id: int
    reply: str
    is_ai_generated: bool
    rag_chunks_used: int
    llm_latency_ms: float
    evaluation_score: float | None
    escalated: bool
    prompt_template: str


# ---------------------------------------------------------------------------
# ServiceNow Dispatcher (preserved from v1.0.0)
# ---------------------------------------------------------------------------

async def send_to_servicenow(ticket_text: str, category: str, urgency: str, sentiment: str):
    """Asynchronously forwards ticket classification metadata to ServiceNow PDI."""
    if not all([SERVICENOW_INSTANCE, SERVICENOW_USER, SERVICENOW_PASS]):
        logger.info("ServiceNow credentials not fully set. Skipping PDI ingestion.")
        return

    instance_url = SERVICENOW_INSTANCE.rstrip("/")
    url = f"{instance_url}/api/{SERVICENOW_SCOPE}/{SERVICENOW_API_ID}/intake"

    payload = {
        "ticket_text": ticket_text,
        "category": category,
        "urgency": urgency,
        "sentiment": sentiment,
    }

    logger.info("ServiceNow Dispatch: Sending ticket payload to %s", url)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                auth=(SERVICENOW_USER, SERVICENOW_PASS),
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=12.0,
            )
            if response.status_code == 201:
                res_data = response.json()
                result = res_data.get("result", {})
                logger.info(
                    "ServiceNow Ingestion Success: Incident %s (sys_id: %s)",
                    result.get("number"),
                    result.get("sys_id"),
                )
            else:
                logger.error(
                    "ServiceNow Ingestion Failure: Status %d",
                    response.status_code,
                )
    except Exception:
        logger.exception("ServiceNow Dispatch Exception")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.api_route("/health", methods=["GET", "HEAD"], tags=["System"])
def health_check():
    """
    Health check — used by Render and Docker HEALTHCHECK.
    Extended in v2.0.0 with component status.
    """
    model_loaded = _classifier is not None
    retriever_ready = _retriever is not None and _retriever.is_ready
    return get_health_status(model_loaded=model_loaded, retriever_ready=retriever_ready)


@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
def serve_frontend(request: Request):
    """Serve the single-page frontend."""
    return templates.TemplateResponse(request, "index.html")


@app.post("/ticket", response_model=TicketResponse, tags=["Tickets"])
def submit_ticket(
    payload: TicketRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Submit a support ticket for classification.

    - **text**: The raw support ticket text (min 5 characters).

    Returns the predicted **category**, **urgency**, **sentiment**, and
    **ml_confidence** score, and stores the ticket in SQLite.

    v2.0.0: Returns ml_confidence and escalated flag.
    """
    text = payload.text.strip()

    # Run ML predictions with confidence
    category, confidence = predict_with_confidence(text, model=_classifier)
    urgency = score_urgency(text)
    sentiment = analyze_sentiment(text)

    # Escalation check
    from app.responsible_ai import should_escalate
    escalated, _ = should_escalate(category, confidence, urgency, sentiment)

    # Log to monitoring
    log_ml_event(category, confidence, urgency, sentiment, len(text))
    log_api_request("/ticket", success=True)
    counters.tickets_total += 1

    # Persist to DB
    ticket = Ticket(
        text=text,
        category=category,
        urgency=urgency,
        sentiment=sentiment,
        ml_confidence=confidence,
        escalated=escalated,
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    logger.info(
        "Ticket #%d: category=%s conf=%.3f urgency=%s sentiment=%s escalated=%s",
        ticket.id, category, confidence, urgency, sentiment, escalated,
    )

    # Queue the ServiceNow PDI ingestion in the background
    background_tasks.add_task(
        send_to_servicenow, text, category, urgency, sentiment
    )

    return TicketResponse(
        id=ticket.id,
        text=ticket.text,
        category=category,
        urgency=urgency,
        sentiment=sentiment,
        ml_confidence=round(confidence, 4),
        escalated=escalated,
    )


@app.post("/ticket/with-image", response_model=TicketResponse, tags=["Tickets"])
async def submit_ticket_with_image(
    background_tasks: BackgroundTasks,
    text: str = Form(..., min_length=5, description="The support ticket text"),
    image: UploadFile = File(..., description="Image of the product, error, or damage"),
    db: Session = Depends(get_db),
):
    """
    Submit a support ticket alongside an image for Computer Vision analysis.
    
    The image is validated, preprocessed (OpenCV), and analyzed (YOLOv8) to detect
    objects relevant to the ticket (e.g., laptops, phones). Detected objects are 
    passed to the AI Orchestrator to improve the final reply.
    """
    text_clean = text.strip()

    # 1. Image Validation
    try:
        file_bytes = await image.read()
        validate_image_file(file_bytes, image.filename, image.content_type)
    except ImageValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read image upload.")

    # 2. Computer Vision Pipeline
    cv_metrics = {}
    detected_objects = []
    has_image = True
    try:
        # Preprocess (OpenCV)
        img_array, prep_lat = load_and_preprocess_image(file_bytes)
        cv_metrics["preprocessing_ms"] = round(prep_lat, 2)
        
        # Detect (YOLOv8)
        detected_objects, inf_lat = cv_detector.detect(img_array)
        cv_metrics["inference_ms"] = round(inf_lat, 2)
        cv_metrics["total_cv_latency_ms"] = round(prep_lat + inf_lat, 2)
        
    except Exception as e:
        logger.error("Computer Vision pipeline failed: %s", e)
        cv_metrics["error"] = str(e)
        # We don't fail the ticket submission if CV fails, we just proceed without CV data

    # 3. Core ML Pipeline
    category, confidence = predict_with_confidence(text_clean, model=_classifier)
    urgency = score_urgency(text_clean)
    sentiment = analyze_sentiment(text_clean)

    # 4. Escalation check
    from app.responsible_ai import should_escalate
    escalated, _ = should_escalate(category, confidence, urgency, sentiment)

    # 5. Log & Persist
    log_ml_event(category, confidence, urgency, sentiment, len(text_clean))
    log_api_request("/ticket/with-image", success=True)
    counters.tickets_total += 1

    ticket = Ticket(
        text=text_clean,
        category=category,
        urgency=urgency,
        sentiment=sentiment,
        ml_confidence=confidence,
        escalated=escalated,
        has_image=has_image,
        cv_metrics=json.dumps(cv_metrics) if cv_metrics else None,
        detected_objects=json.dumps(detected_objects) if detected_objects else None,
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    logger.info(
        "TicketWithImage #%d: category=%s conf=%.3f objects=%d CV_lat=%.1fms",
        ticket.id, category, confidence, len(detected_objects), cv_metrics.get("total_cv_latency_ms", 0.0)
    )

    background_tasks.add_task(send_to_servicenow, text_clean, category, urgency, sentiment)

    return TicketResponse(
        id=ticket.id,
        text=ticket.text,
        category=category,
        urgency=urgency,
        sentiment=sentiment,
        ml_confidence=round(confidence, 4),
        escalated=escalated,
    )


@app.post("/ticket/reply", response_model=ReplyResponse, tags=["Tickets"])
def get_ticket_reply(payload: ReplyRequest, db: Session = Depends(get_db)):
    """
    Generate an AI-drafted reply for a previously submitted ticket.

    v2.0.0: Uses full AI orchestration pipeline:
      - RAG retrieval from knowledge base
      - Structured prompt engineering
      - Groq LLM (llama-3.1-8b-instant)
      - Response evaluation (heuristic quality checks)
      - One regeneration attempt if quality below threshold
      - Fallback to static template if LLM unavailable

    - **ticket_id**: The ID returned by POST /ticket.
    """
    ticket = db.query(Ticket).filter(Ticket.id == payload.ticket_id).first()
    if not ticket:
        raise HTTPException(
            status_code=404,
            detail=f"Ticket with id={payload.ticket_id} not found.",
        )

    log_api_request("/ticket/reply", success=True)

    # Run the full AI orchestration pipeline
    
    # Parse CV objects if present
    cv_objects = None
    if ticket.detected_objects:
        import json
        try:
            cv_objects = json.loads(ticket.detected_objects)
        except Exception:
            pass

    result = orchestrate(
        ticket_text=ticket.text,
        category=ticket.category,
        confidence=ticket.ml_confidence or 0.5,
        urgency=ticket.urgency,
        sentiment=ticket.sentiment,
        retriever=_retriever,
        cv_objects=cv_objects,
    )

    # Persist reply and metrics back to the ticket record
    ticket.reply = result.reply
    ticket.is_ai_reply = result.is_ai_generated
    ticket.rag_chunks_used = result.rag_chunks_used
    ticket.llm_latency_ms = result.llm_latency_ms
    ticket.evaluation_score = result.evaluation.quality_score if result.evaluation else None
    ticket.escalated = result.escalated
    db.commit()

    logger.info(
        "Reply #%d: ai=%s rag_chunks=%d latency=%.1fms eval=%.3f escalated=%s",
        ticket.id,
        result.is_ai_generated,
        result.rag_chunks_used,
        result.llm_latency_ms,
        result.evaluation.quality_score if result.evaluation else 0.0,
        result.escalated,
    )

    return ReplyResponse(
        ticket_id=ticket.id,
        reply=result.reply,
        is_ai_generated=result.is_ai_generated,
        rag_chunks_used=result.rag_chunks_used,
        llm_latency_ms=result.llm_latency_ms,
        evaluation_score=result.evaluation.quality_score if result.evaluation else None,
        escalated=result.escalated,
        prompt_template=result.prompt_template,
    )


# ---------------------------------------------------------------------------
# Analytics & Metrics Endpoints
# ---------------------------------------------------------------------------

@app.get("/analytics", tags=["Analytics"])
def full_analytics(db: Session = Depends(get_db)):
    """
    Full analytics dashboard.

    Returns:
      - Ticket distribution (from database — persistent)
      - Session-level AI/LLM performance counters (resets on restart)
      - ML model evaluation metrics (from training run)
    """
    return get_full_analytics(db)


@app.get("/analytics/tickets", tags=["Analytics"])
def ticket_analytics(db: Session = Depends(get_db)):
    """
    Ticket distribution statistics from the database.

    Returns category, urgency, sentiment distributions and AI reply stats.
    These persist across application restarts.
    """
    return get_ticket_stats(db)


@app.get("/analytics/ai", tags=["Analytics"])
def ai_analytics():
    """
    AI/LLM session performance statistics.

    Returns LLM call counts, success rates, average latency, RAG stats,
    and escalation counts. Resets on application restart.
    """
    return get_ai_stats()


@app.get("/metrics", tags=["Analytics"])
def ml_metrics():
    """
    ML model evaluation metrics from the last training run.

    Returns accuracy, per-class F1 scores, confusion matrix, and
    cross-validation scores. Computed from actual training execution.
    """
    return get_model_metrics()
