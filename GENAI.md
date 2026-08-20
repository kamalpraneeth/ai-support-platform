# AI Support Platform — Generative AI (GenAI)

The platform leverages Large Language Models (LLMs) to construct human-readable, helpful, and empathetic responses to customer support tickets.

## 1. The Model

We use the **Groq API** calling the **LLaMA-3.1-8b-instant** model.
- **Why LLaMA 3.1 8B?** It strikes an ideal balance between reasoning capability, instruction following, and extreme speed.
- **Why Groq?** Groq's LPU architecture provides ultra-low latency inference, generating hundreds of tokens per second. This ensures that the customer support API feels instantaneous.

## 2. Prompt Engineering (`app/prompts/`)

The GenAI pipeline heavily relies on structured prompt engineering to control the LLM's output and enforce platform rules.

### Prompt Templates (`templates.py`)
Templates are implemented as frozen dataclasses (`PromptTemplate`). We currently maintain two versions:
- `SUPPORT_REPLY_V1`: The primary prompt used when RAG chunks are successfully retrieved.
- `SUPPORT_REPLY_FALLBACK_V1`: Used when no relevant knowledge base articles are found.

**Key Elements of the System Prompt:**
1. **Persona**: Defines the assistant as a helpful, professional customer support agent.
2. **Context**: Instructs the LLM to use the provided Knowledge Base articles.
3. **Safety Rules**: Strict prohibitions against fabricating links, phone numbers, or account details.
4. **Tone Enforcement**: Modulates tone based on the ML-predicted Sentiment and Urgency (e.g., being more empathetic for negative sentiment, and explicitly prioritizing high urgency tickets).

### Prompt Builder (`builder.py`)
The prompt builder dynamically constructs the final prompt payload (`PromptPayload`) by:
1. Injecting the original ticket text.
2. Injecting the ML metadata (Category, Sentiment, Urgency).
3. Formatting and appending the retrieved RAG chunks.
4. Returning a standardized list of messages (System and User) ready for the Groq API.

## 3. Fallback Mechanism

If the Groq API key is missing (e.g., running locally without credentials), or if the LLM API fails, the `Orchestrator` automatically degrades gracefully. It falls back to a deterministic, canned response apologizing for the delay and promising human follow-up. This ensures the API always returns a `200 OK` and never drops a ticket.
