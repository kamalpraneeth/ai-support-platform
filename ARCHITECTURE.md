# AI Support Platform — Architecture

## Overview

The AI Customer Support Platform is a production-grade FastAPI application that combines Machine Learning (ML) and Generative AI (GenAI) to automate and enhance customer support workflows. The system automatically categorizes incoming support tickets, determines their urgency and sentiment, retrieves relevant knowledge base articles via Retrieval-Augmented Generation (RAG), and generates high-quality, grounded responses using Groq's LLaMA 3.1 LLM.

## Core Components

The architecture is built on a modular, decoupled design, ensuring scalability, maintainability, and clear separation of concerns.

### 1. API Layer (`app/main.py`)
- Built with **FastAPI** for high performance and automatic OpenAPI documentation.
- Handles incoming HTTP requests for ticket submission, reply generation, health checks, and analytics.
- Manages application lifecycle events (lifespan hook) to initialize ML models and RAG index into memory on startup.

### 2. Data Engineering & Persistence (`app/data_pipeline.py`, `app/models.py`, `app/database.py`)
- Uses **SQLAlchemy** with SQLite for lightweight, transactional persistence of tickets and their metadata.
- Records all AI-generated metrics (confidence, RAG chunks used, LLM latency) for auditing.
- `data_pipeline.py` provides validation, normalization, and deduplication of training datasets.

### 3. ML Pipeline (`app/ml/`)
- A traditional machine learning pipeline using **scikit-learn** (TF-IDF + Logistic Regression).
- Responsible for multi-class classification (Billing, Technical, Account, General).
- Provides confidence scoring used by the orchestration layer to trigger human escalation.
- Rule-based heuristics (via `VADER`) are used for fast, deterministic Urgency and Sentiment scoring.

### 4. RAG Pipeline (`app/rag/`)
- **Knowledge Base (`app/rag/knowledge_base.py`)**: Loads curated domain knowledge from JSON files.
- **Retriever (`app/rag/retriever.py`)**: Uses a TF-IDF vectorizer and cosine similarity to retrieve relevant chunks based on the user's query. Implements **Category Boosting** to rank chunks matching the predicted ML category higher.

### 5. Generative AI & Prompt Engineering (`app/prompts/`)
- Uses **Groq API** (LLaMA-3.1-8b-instant) for fast inference.
- Implements versioned prompt templates (`templates.py`).
- dynamically constructs prompt payloads (`builder.py`) by injecting ticket details, ML predictions (category, sentiment, urgency), and retrieved RAG chunks.

### 6. AI Orchestrator (`app/ai_orchestrator.py`)
- The central brain of the platform.
- Coordinates the flow: Validation -> RAG Retrieval -> Prompt Construction -> LLM Invocation -> Response Evaluation -> Escalation checks.
- Returns a unified `OrchestratorResult` containing the generated reply and execution metadata.

### 7. Evaluation & Responsible AI (`app/evaluation.py`, `app/responsible_ai.py`)
- **Evaluation**: Performs post-generation heuristic checks (relevance, completeness, groundedness, safety, sign-off) to score the generated response and detect hallucinations or poor quality.
- **Responsible AI**: Pre-generation input validation (prompt injection detection, PII flagging) and post-generation output validation (checking for fabricated sensitive data). Determines if a ticket requires human escalation based on urgency and ML confidence.

### 8. Monitoring & Analytics (`app/monitoring.py`, `app/analytics.py`)
- Tracks application usage, token counts, error rates, and API latency.
- Provides endpoints for dashboards to aggregate ticket volume by category, sentiment trends, and AI session statistics.

## Flow Diagram

1. **User Submission**: `POST /ticket` -> API stores ticket, returns ID.
2. **AI Reply Request**: `POST /ticket/reply` -> API loads ticket.
3. **ML Classification**: Classifier predicts Category (with confidence), Urgency, Sentiment.
4. **Input Validation**: Responsible AI checks for prompt injection.
5. **RAG Retrieval**: Retrieve relevant KB chunks based on ticket text + predicted Category.
6. **Prompt Building**: Construct prompt with RAG chunks and ticket data.
7. **LLM Generation**: Groq generates response.
8. **Evaluation**: Response is checked for completeness, relevance, and safety.
9. **Escalation Check**: If confidence is low, or response fails evaluation, escalate to human.
10. **Response**: Final reply and metadata returned to user.
