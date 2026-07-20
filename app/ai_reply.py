"""
AI Reply Generator: uses Groq's API (llama-3.1-8b-instant model) to draft
a professional customer support reply for a given ticket.

Fallback behaviour:
  If GROQ_API_KEY is not set, or if the API call fails for any reason
  (network error, rate limit, etc.), a static template reply is returned.
  This ensures the /ticket/reply endpoint always returns something useful.

Why Groq?
- Free tier with generous rate limits (no credit card needed to start).
- OpenAI-compatible API — swapping to OpenAI or Gemini is a one-line change.
- Fast inference (LPU hardware) — typically <1s for short prompts.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# System prompt keeps replies professional, concise, and empathetic
SYSTEM_PROMPT = (
    "You are a helpful and empathetic customer support agent. "
    "When given a customer support ticket, write a professional, concise reply "
    "(3-5 sentences). Acknowledge the issue, apologize if appropriate, "
    "and tell the customer what the next step is. "
    "Do not make up specific details (account numbers, dates, etc.). "
    "Sign off as 'Support Team'."
)

FALLBACK_REPLY = (
    "Thank you for reaching out to us. We have received your ticket and our "
    "support team is looking into this as a priority. You can expect a "
    "detailed response within 24 hours. If this is urgent, please reply to "
    "this message and we will escalate accordingly.\n\n"
    "Best regards,\nSupport Team"
)


def generate_reply(ticket_text: str) -> tuple[str, bool]:
    """
    Generate a draft reply for the given support ticket.

    Returns:
        (reply_text, is_ai_generated) — the second element tells the caller
        whether the reply came from the AI model or the fallback template.
    """
    api_key: Optional[str] = os.getenv("GROQ_API_KEY")

    if not api_key:
        logger.warning(
            "GROQ_API_KEY not set — returning fallback reply. "
            "Add GROQ_API_KEY to your .env file to enable AI-generated replies."
        )
        return FALLBACK_REPLY, False

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Customer ticket:\n{ticket_text}"},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        logger.info("Groq reply generated successfully (%d chars).", len(reply))
        return reply, True

    except Exception as exc:
        # Log exception type so auth/network/rate-limit errors are distinguishable
        logger.error(
            "Groq API call failed [%s]: %s — returning fallback reply.",
            type(exc).__name__,
            exc,
        )
        return FALLBACK_REPLY, False
