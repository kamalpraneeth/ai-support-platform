# AI Customer Support Platform

An end-to-end AI-powered customer support platform combining machine learning, Generative AI, RAG, computer vision, responsible AI, automated evaluation, and human escalation.

## Core Capabilities
├── Machine Learning Classification
├── Data Engineering & Preprocessing
├── Generative AI / LLM
├── Prompt Engineering
├── RAG
├── Conversational Chatbot
├── Computer Vision (YOLOv8)
├── Optical Character Recognition (easyocr)
├── Responsible AI
├── Response Evaluation
├── Human Escalation
├── REST APIs
├── Analytics & Monitoring
├── Docker
├── CI/CD
└── Automated Testing

## 📚 Documentation

The architecture and implementation details are thoroughly documented in the following guides:

1. **[Architecture Overview](ARCHITECTURE.md)** — Core components and data flow.
2. **[API Reference](API.md)** — REST API endpoints and schemas.
3. **[ML Pipeline](ML_PIPELINE.md)** — Data engineering, TF-IDF + Logistic Regression classification, and heuristic scoring.
4. **[Generative AI](GENAI.md)** — LLaMA 3.1 LLM integration and prompt engineering.
5. **[RAG Pipeline](RAG.md)** — Knowledge base, TF-IDF retrieval, and Category Boosting.
6. **[Response Evaluation](EVALUATION.md)** — Post-generation heuristic checks (completeness, groundedness, relevance).
7. **[Responsible AI](RESPONSIBLE_AI.md)** — Prompt injection defense, PII flagging, output safety validation, and Human-in-the-Loop escalation logic.
8. **[Monitoring & Analytics](MONITORING.md)** — Structured logging, internal counters, and business analytics.
9. **[Deployment & CI](DEPLOYMENT.md)** — Docker usage, environment variables, and GitHub Actions CI.
10. **[Computer Vision](COMPUTER_VISION.md)** — Optional OpenCV + YOLOv8 integration for analyzing image uploads.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- [Groq API Key](https://console.groq.com/keys) (required for AI generation; without it, the app safely falls back to canned responses).

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/kamalpraneeth/ai-support-platform.git
cd ai-support-platform

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Copy the `.env.example` file to `.env` and configure your API key.
```bash
cp .env.example .env
```
Edit `.env` and set `GROQ_API_KEY=your_actual_api_key`.

### 4. Run the Application
```bash
uvicorn app.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`. You can view the interactive OpenAPI docs at `http://127.0.0.1:8000/docs`.

### 5. Run the Test Suite
The project includes a comprehensive 185+ test suite covering ML, API, RAG, Computer Vision, and Responsible AI.
```bash
pytest tests/ -v
```

## 🏗️ Architecture Highlight

The core orchestration flow for a ticket is:
`Image Processing (Optional) -> Validation -> ML Classification -> RAG Retrieval -> LLM Generation -> Output Evaluation -> Escalation Check`.

By combining fast, deterministic ML for routing and strict Responsible AI rules for safety, we ensure that the LLM is only utilized when appropriate and safe.
