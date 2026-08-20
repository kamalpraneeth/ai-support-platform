# AI Support Platform — Responsible AI

The Responsible AI module (`app/responsible_ai.py`) acts as the primary defense mechanism against malicious inputs, data leaks, and high-risk automated actions. It sits as a middleware layer in the `Orchestrator`, intercepting data before it reaches the LLM and before responses reach the user.

## 1. Input Validation (Pre-Generation)

Before a ticket is processed by the LLM, the system runs strict validation checks.

### Prompt Injection Detection
Malicious users may attempt to subvert the LLM using "jailbreak" or "ignore previous instructions" attacks to extract system prompts or manipulate behavior.
- The system scans the ticket text for known attack signatures using regex patterns (`ignore all previous`, `jailbreak`, `you are now`, etc.).
- **Action**: If an injection is detected, the request is immediately rejected at the API level (400 Bad Request) and the LLM is never invoked.

### Length & Resource Exhaustion
To prevent Denial of Wallet (DoW) attacks or context-window overflow:
- Inputs over `5000` characters are rejected entirely.
- Inputs between `2000` and `5000` characters are processed but flagged as `is_suspiciously_long`.

### PII Detection
The system scans for Personal Identifiable Information (Emails, Phone numbers) in the incoming ticket.
- **Action**: Finding PII does *not* reject the ticket (since users often include this in support requests), but it logs the detection event in the monitoring counters for compliance tracking.

## 2. Output Validation (Post-Generation)

After the LLM generates a response, but before it is returned to the user, the output is validated.

### Hallucination & Fabrication Safeguards
Even with strict prompt rules, LLMs can hallucinate sensitive data. The output validator specifically scans the LLM's response for:
- 10+ digit sequences (fabricated account numbers, SSNs, credit cards).
- URLs that do not belong to the company domain.
- Phone numbers that are not the official support line.
- **Action**: If fabricated data is found, the response fails validation and triggers the escalation fallback.

## 3. Human-in-the-Loop Escalation Logic

The most critical Responsible AI feature is knowing when the AI should step aside. The `should_escalate()` function implements a deterministic rules engine to route tickets to human agents rather than relying on AI generation.

**Escalation Triggers:**
1. **Low ML Confidence**: If the ML classifier is unsure of the category (confidence < `0.65`), the AI defers to a human rather than risking an incorrect RAG retrieval.
2. **High Risk Sentiments**: If the user is both highly angry (`Sentiment = Negative`) and has a severe issue (`Urgency = High`), the AI defers to human empathy to prevent further frustration.
3. **Sensitive Categories**: Certain categories (like `Account` suspensions or security issues) combined with `High` urgency always escalate.

**Result of Escalation**:
When escalated, the API returns `escalated = True` and replaces the LLM generation with a standard, safe fallback message assuring the user that a human agent is reviewing their case.
