# AI Support Platform — Response Evaluation

To ensure high-quality and safe AI outputs, the platform includes a post-generation Evaluation module (`app/evaluation.py`). 

> **Important**: This evaluation relies on strict, deterministic heuristics rather than a slow, expensive LLM-as-a-Judge architecture. We do not claim this is a guaranteed "hallucination detector"; rather, it is a fast safeguard designed to catch common generative errors before the user sees them.

## 1. Evaluation Checks

Every LLM response is scored against five heuristic checks:

### A. Completeness Check
Verifies that the response is of a sufficient length (`MIN_RESPONSE_LENGTH`) and includes actionable next steps. It scans for action-oriented keywords (e.g., "please", "contact", "reply", "click", "visit").

### B. Relevance Check
Calculates the word overlap between the generated response and the original user ticket text (excluding common stop words). If the response shares almost no vocabulary with the user's inquiry, it scores poorly, indicating the LLM may have generated an off-topic reply.

### C. Groundedness Check
If RAG chunks were provided to the LLM, this check calculates the term overlap between the generated response and the provided KB chunks. A high overlap indicates the LLM successfully grounded its answer in the provided knowledge, rather than fabricating its own policy. If no chunks were provided, this check automatically passes.

### D. Safety Check
Scans the response for patterns indicating severe fabrication or PII leaks. It uses regex patterns to detect:
- Fabricated long digit sequences (simulated account numbers or SSNs).
- Unapproved external URLs (phishing risk).
- Unapproved phone numbers (the LLM hallucinating a call center number).

### E. Sign-off Check
Verifies that the LLM concluded the message professionally, looking for standard signatures (e.g., "Best regards", "Support Team").

## 2. Quality Scoring & Regeneration

The results of the above checks are aggregated into a `quality_score` (0.0 to 1.0). 
- If the score falls below a critical threshold (`MIN_QUALITY_SCORE = 0.5`), or if the Safety Check fails entirely, the `needs_regeneration` flag is set to `True`.
- Currently, if an evaluation fails, the `Orchestrator` flags the ticket for human escalation (`escalated = True`) to prevent sending a sub-par AI response to the user.
